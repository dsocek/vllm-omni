# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CausalForcingPipeline — vllm-omni pipeline for thu-ml/Causal-Forcing.

Wan2.1-T2V-1.3B-based causal / autoregressive few-step video diffusion. Video
latents are generated block-by-block (``num_frame_per_block``) with a
dict-based KV cache so new frames attend causally to already-generated history.
The ``framewise-1step`` variant does a single denoise step per block.

Runs under the base ``DiffusionEngine`` (no AR-Diffusion engine): the KV cache
is owned by this pipeline. See :mod:`.causal_wan_model` for the transformer and
:mod:`.scheduling_causal_forcing` for the flow-matching scheduler.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn as nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.hub_prefetch import (
    from_pretrained_with_prefetch,
    prefetch_subfolders,
)
from vllm_omni.diffusion.models.causal_forcing.causal_wan_model import CausalWanModel
from vllm_omni.diffusion.models.causal_forcing.scheduling_causal_forcing import FlowMatchScheduler
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = logging.getLogger(__name__)

# Default WAN 1.3B negative prompt (unused when CFG is off, kept for parity).
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# framewise-1step defaults (configs/causal_forcing_dmd_framewise_1step.yaml).
DEFAULT_DENOISING_STEP_LIST = [1000]
DEFAULT_DENOISING_STEP_LIST_FIRST_CHUNK = [1000, 750, 500, 250]
DEFAULT_TIMESTEP_SHIFT = 5.0
DEFAULT_CONTEXT_NOISE = 0
DEFAULT_NUM_FRAME_PER_BLOCK = 1

# 1.3B transformer geometry (Wan2.1-T2V-1.3B).
TRANSFORMER_CONFIG = {
    "model_type": "t2v",
    "patch_size": (1, 2, 2),
    "text_len": 512,
    "in_dim": 16,
    "dim": 1536,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 12,
    "num_layers": 30,
    "local_attn_size": -1,
    "sink_size": 0,
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}

# Global-attention KV pool size (upstream default for local_attn_size == -1).
GLOBAL_KV_CACHE_SIZE = 32760


def _read_model_index(model: str) -> dict:
    """Read the assembled model's ``model_index.json`` inference constants.

    The engine reads that file only for ``_class_name``, so on the offline path
    the checkpoint's own constants never reach ``od_config.model_config``. Best
    effort: a missing or malformed file just means "no overrides".
    """
    path = os.path.join(model, "model_index.json")
    try:
        with open(path) as f:
            index = json.load(f)
    except (OSError, ValueError) as e:
        logger.debug("CausalForcing: no usable model_index.json at %s (%s)", path, e)
        return {}
    return index if isinstance(index, dict) else {}


class CausalForcingPipeline(nn.Module, ProgressBarMixin, SupportsComponentDiscovery):
    """Causal-Forcing text-to-video pipeline (few-step, KV-cached, base engine).

    Supports three ``_stage_role`` modes so the pipeline can run either
    aggregated (default) or disaggregated across two stages:

    - ``"full"``  — text encode + DiT rollout + VAE decode in one stage (the
      original monolithic behavior; used by the single-stage deploy).
    - ``"dit"``   — text encoder + transformer only. ``forward`` runs the
      rollout and emits the raw denoised latents ([B, C, F, H, W]); the VAE is
      NOT built. Used as stage 0 of the disaggregated pipeline.
    - ``"vae"``   — VAE only. ``forward`` reads latents handed over from the
      DiT stage (``prompt["extra"]["latents"]``) and decodes them to pixels;
      the transformer / text encoder are NOT built. Used as stage 1.

    The role is a class attribute set by the ``CausalForcingDiTPipeline`` /
    ``CausalForcingVAEPipeline`` subclasses registered as distinct model archs;
    the base class stays ``"full"`` so existing single-stage runs are unchanged.
    """

    _stage_role: ClassVar[str] = "full"

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        # Which components this stage owns. "full" builds everything; the
        # disaggregated roles build only what they run so the DiT stage and the
        # VAE stage each fit comfortably on a single card.
        self._build_dit = self._stage_role in ("full", "dit")
        self._build_vae = self._stage_role in ("full", "vae")
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)
        # Inference constants come from the model's own ``model_index.json``,
        # with deploy/engine ``model_config`` overriding per key. The engine reads
        # that file only for ``_class_name`` and never populates
        # ``od_config.model_config`` on the offline path, so without this merge the
        # checkpoint's constants (notably ``local_attn_size``) are unreachable
        # there — and a partial engine-side ``model_config`` must not silently
        # drop the rest of them.
        model_config = {**_read_model_index(model), **(od_config.model_config or {})}

        # ---- Inference constants (model_index.json / model_config overrides) ----
        self.num_frame_per_block = int(model_config.get("num_frame_per_block", DEFAULT_NUM_FRAME_PER_BLOCK))
        self.context_noise = int(model_config.get("context_noise", DEFAULT_CONTEXT_NOISE))
        self.negative_prompt = model_config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        timestep_shift = float(model_config.get("timestep_shift", DEFAULT_TIMESTEP_SHIFT))
        denoising_step_list = model_config.get("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)
        denoising_step_list_first_chunk = model_config.get(
            "denoising_step_list_first_chunk", DEFAULT_DENOISING_STEP_LIST_FIRST_CHUNK
        )
        # Sliding-window causal attention. ``-1`` means full global attention,
        # which caps a clip at ``GLOBAL_KV_CACHE_SIZE`` tokens (21 latent frames
        # at 480x832); a finite window gives constant KV memory and a flat
        # per-latent cost, which is what streaming beyond that cap needs.
        # ``CF_LOCAL_ATTN_SIZE`` is a debug/benchmark escape hatch only.
        local_attn_size = int(model_config.get("local_attn_size", TRANSFORMER_CONFIG["local_attn_size"]))
        _env_attn = os.environ.get("CF_LOCAL_ATTN_SIZE")
        if _env_attn is not None:
            local_attn_size = int(_env_attn)
            logger.info("CausalForcing: local_attn_size=%d (CF_LOCAL_ATTN_SIZE override)", local_attn_size)

        # ---- Scheduler + warped denoising schedules ----
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=1000,
            shift=timestep_shift,
            sigma_max=1.0,
            sigma_min=0.0,
            extra_one_step=True,
        )
        # warp_denoising_step=true in the framewise config.
        self.denoising_step_list = self.scheduler.warp_denoising_step(denoising_step_list)
        self.denoising_step_list_first_chunk = (
            self.scheduler.warp_denoising_step(denoising_step_list_first_chunk)
            if denoising_step_list_first_chunk is not None
            else None
        )

        # ---- Components (Wan2.1-T2V-1.3B), built per stage role ----
        # DiT stage owns tokenizer + text encoder + transformer; VAE stage owns
        # only the VAE. "full" owns everything. Only prefetch what we build so a
        # role-specific stage never touches the other component's weights.
        component_subfolders = []
        if self._build_dit:
            component_subfolders += ["tokenizer", "text_encoder"]
        if self._build_vae:
            component_subfolders += ["vae"]
        prefetch_subfolders(model, component_subfolders, local_files_only=local_files_only)

        self.tokenizer = None
        self.text_encoder = None
        self.transformer = None
        self.vae = None
        self.weights_sources: list = []

        if self._build_dit:
            self.tokenizer = from_pretrained_with_prefetch(
                AutoTokenizer.from_pretrained,
                model,
                subfolder="tokenizer",
                prefetch_list=component_subfolders,
                local_files_only=local_files_only,
            )
            self.text_encoder = from_pretrained_with_prefetch(
                UMT5EncoderModel.from_pretrained,
                model,
                subfolder="text_encoder",
                prefetch_list=component_subfolders,
                local_files_only=local_files_only,
                torch_dtype=self.dtype,
            ).to(self.device)

            # ---- Transformer (weights loaded eagerly from the .pt below) ----
            transformer_kwargs = dict(TRANSFORMER_CONFIG)
            transformer_kwargs["local_attn_size"] = local_attn_size
            transformer_kwargs["num_frame_per_block"] = self.num_frame_per_block
            self.transformer = CausalWanModel(**transformer_kwargs).to(device=self.device, dtype=self.dtype)
            self.transformer.eval()

            # The checkpoint is a single-key .pt (not diffusers-shaped): load
            # eagerly and let ``load_weights`` be a no-op for the standard loader.
            ckpt_name = model_config.get("transformer_checkpoint", "framewise-1step.pt")
            ckpt_path = ckpt_name if os.path.isabs(ckpt_name) else os.path.join(model, ckpt_name)
            self._load_transformer_checkpoint(ckpt_path)

        if self._build_vae:
            self.vae = from_pretrained_with_prefetch(
                DistributedAutoencoderKLWan.from_pretrained,
                model,
                subfolder="vae",
                prefetch_list=component_subfolders,
                local_files_only=local_files_only,
                torch_dtype=torch.float32,
            ).to(self.device)
            self.register_buffer(
                "vae_latents_mean",
                torch.tensor(self.vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "vae_latents_inv_std",
                (1.0 / torch.tensor(self.vae.config.latents_std, dtype=torch.float32)).view(1, -1, 1, 1, 1),
                persistent=False,
            )

    # -----------------------------------------------------------------------
    # torch.compile setup
    # -----------------------------------------------------------------------

    def setup_compile(self) -> None:
        """Compile the DiT blocks and the VAE decoder for inference.

        Called by the model runner when ``enforce_eager`` is False (instead of the
        default transformer-only compile). Each target is compiled independently so
        a failure in one (e.g. DiT inductor codegen) never disables the others —
        notably the VAE decoder, which is the decode-throughput bottleneck.

        The VAE decoder is compiled at ``decoder.forward`` (a single latent frame),
        NOT ``_decode``/streaming loop (which is a Python frame loop). This matches
        the per-frame streaming decode path and keeps the temporal feat_cache eager.

        The one-time inductor codegen cost is paid once at load and amortizes to
        zero over a long-lived serving process.
        """
        from vllm_omni.diffusion.compile import regionally_compile

        compile_kwargs = {"dynamic": True}

        # DiT: regional per-block compile (uses _repeated_blocks on the transformer).
        if self.transformer is not None:
            try:
                self.transformer = regionally_compile(self.transformer, **compile_kwargs)
                logger.info("CausalForcing: DiT transformer compiled (regional, per-block).")
            except Exception as exc:  # noqa: BLE001 - never let compile break serving
                logger.warning("CausalForcing: DiT torch.compile failed (%s); DiT stays eager.", exc)

        # VAE decoder: the decode-throughput bottleneck. Compile decoder.forward
        # (per-latent-frame call); the streaming loop and feat_cache stay eager.
        if self.vae is not None:
            try:
                self.vae.decoder.forward = torch.compile(self.vae.decoder.forward, **compile_kwargs)
                logger.info("CausalForcing: VAE decoder compiled (decoder.forward).")
            except Exception as exc:  # noqa: BLE001
                logger.warning("CausalForcing: VAE torch.compile failed (%s); VAE stays eager.", exc)

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def _load_transformer_checkpoint(self, ckpt_path: str) -> None:
        """Load the ``generator_ema`` state dict into the transformer with remap."""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Causal-Forcing transformer checkpoint not found at {ckpt_path!r}. "
                "Run examples/offline_inference/causal_forcing/download_causal_forcing.py first."
            )
        logger.info("CausalForcing: loading transformer weights from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # framewise checkpoints store the EMA generator under "generator_ema";
        # fall back to "generator" or a bare state dict.
        if isinstance(ckpt, dict) and "generator_ema" in ckpt:
            state_dict = ckpt["generator_ema"]
        elif isinstance(ckpt, dict) and "generator" in ckpt:
            state_dict = ckpt["generator"]
        else:
            state_dict = ckpt

        remapped = [(self._strip_prefix(name), tensor) for name, tensor in state_dict.items()]
        loaded = self._apply_transformer_weights(iter(remapped))
        # Self-attn q/k/v (weight+bias) fuse into a single qkv param, so the count
        # of loaded param names is less than the checkpoint tensor count. Verify
        # coverage by the set of expected destination param names instead.
        expected = {self._fuse_qkv_name(name) for name, _ in remapped}
        missing = expected - loaded
        if missing:
            logger.warning(
                "CausalForcing: %d/%d checkpoint tensors mapped to %d params; %d expected params unloaded: %s",
                len(state_dict),
                len(state_dict),
                len(loaded),
                len(missing),
                sorted(missing)[:8],
            )
        else:
            logger.info("CausalForcing: loaded all %d transformer tensors (%d params)", len(state_dict), len(loaded))

    @staticmethod
    def _strip_prefix(name: str) -> str:
        """Strip the FSDP wrapper prefix from a checkpoint key."""
        prefix = "model._fsdp_wrapped_module."
        if name.startswith(prefix):
            name = name[len(prefix) :]
        return "transformer." + name

    @staticmethod
    def _fuse_qkv_name(name: str) -> str:
        """Map a remapped self-attn q/k/v key to its fused ``qkv`` param name."""
        for shard_id in ("q", "k", "v"):
            needle = f".self_attn.{shard_id}."
            if needle in name:
                return name.replace(needle, ".self_attn.qkv.")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Engine loader entry point.

        All weights are loaded eagerly in ``__init__`` (the transformer from the
        ``.pt`` via :meth:`_apply_transformer_weights`, and the text encoder / VAE
        via ``from_pretrained``), so ``self.weights_sources`` is empty and the
        loader passes no tensors here. Report every initialized parameter name so
        the loader's strict "not initialized from checkpoint" check is satisfied.
        Any tensors that *are* passed (defensive) are still applied.
        """
        loaded = self._apply_transformer_weights(weights)
        loaded |= {name for name, _ in self.named_parameters()}
        loaded |= {name for name, _ in self.named_buffers()}
        return loaded

    def _apply_transformer_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Apply remapped ``transformer.*`` weights, fusing self-attn q/k/v.

        The checkpoint has separate ``self_attn.{q,k,v}`` — route each to the
        packed :class:`QKVParallelLinear` ``qkv`` param with its shard id.
        Cross-attn q/k/v stay separate; ffn.0/ffn.2 map to the Sequential indices.
        """
        loaded: set[str] = set()
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())

        for name, tensor in weights:
            new_name = name
            qkv_shard_id: str | None = None
            for shard_id in ("q", "k", "v"):
                needle = f".self_attn.{shard_id}."
                if needle in new_name:
                    new_name = self._fuse_qkv_name(new_name)
                    qkv_shard_id = shard_id
                    break

            if new_name in params:
                param = params[new_name]
                if qkv_shard_id is not None:
                    param.weight_loader(param, tensor, qkv_shard_id)
                else:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor)
                loaded.add(new_name)
            elif new_name in buffers:
                buffers[new_name].data.copy_(tensor)
                loaded.add(new_name)
            else:
                logger.debug("CausalForcing: unmapped checkpoint key %s", name)

        return loaded

    # -----------------------------------------------------------------------
    # Prompt encoding
    # -----------------------------------------------------------------------

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def encode_prompt(self, prompt: str, max_sequence_length: int = 512) -> torch.Tensor:
        """Encode a prompt to a [1, text_len, text_dim] embedding (zero-padded)."""
        text_inputs = self.tokenizer(
            [self._prompt_clean(prompt)],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        # Feed the encoder on ITS device (the engine may place it on CPU/device
        # independently of self.device), then bring embeddings to the compute device.
        enc_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(ids.to(enc_device), mask.to(enc_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        # Zero the padding positions, then pad back to exactly max_sequence_length.
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )
        return prompt_embeds

    # -----------------------------------------------------------------------
    # KV cache (dict-based, owned by the pipeline)
    # -----------------------------------------------------------------------

    def _init_caches(
        self, batch_size: int, dtype: torch.dtype, device: torch.device, kv_cache_size: int
    ) -> tuple[list[dict], list[dict]]:
        n = self.transformer.num_layers
        head_dim = self.transformer.dim // self.transformer.num_heads
        tp_num_heads = getattr(self.transformer.blocks[0].self_attn, "tp_num_heads", self.transformer.num_heads)
        kv_cache = [
            {
                "k": torch.zeros([batch_size, kv_cache_size, tp_num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, tp_num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            for _ in range(n)
        ]
        crossattn_cache = [
            {
                "k": torch.zeros([batch_size, 512, tp_num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, tp_num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False,
            }
            for _ in range(n)
        ]
        return kv_cache, crossattn_cache

    # -----------------------------------------------------------------------
    # Forward (framewise rollout)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        # VAE stage of the disaggregated pipeline: no rollout, just decode the
        # latents handed over from the DiT stage.
        if self._stage_role == "vae":
            return self._forward_vae_decode(req)

        if len(req.prompts) != 1:
            raise ValueError("CausalForcingPipeline supports a single prompt per request.")
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        if not prompt:
            raise ValueError("Prompt is required for Causal-Forcing generation.")

        sp = req.sampling_params
        height = sp.height or 480
        width = sp.width or 832
        num_frames = sp.num_frames or 81
        output_type = sp.output_type or "np"
        # DiT stage of the disaggregated pipeline emits raw model-space latents;
        # the downstream VAE stage owns denorm + decode. Force the latent path
        # (no VAE work here — the VAE isn't even built in this role).
        if self._stage_role == "dit":
            output_type = "latent"
        # Benchmark knob: emit raw latents and skip VAE decode entirely (no
        # denorm, no decoder) so a run measures pure DiT rollout throughput.
        if os.environ.get("CF_DIT_ONLY") == "1":
            output_type = "latent"

        # Latent geometry: VAE downsamples 8x spatial, 4x temporal (+1).
        latent_h = height // 8
        latent_w = width // 8
        latent_frames = (num_frames - 1) // 4 + 1
        frame_seqlen = (latent_h // self.transformer.patch_size[1]) * (latent_w // self.transformer.patch_size[2])

        device = self.device
        dtype = self.dtype
        batch_size = 1
        nfpb = self.num_frame_per_block
        if latent_frames % nfpb != 0:
            raise ValueError(f"latent frames {latent_frames} must be divisible by num_frame_per_block {nfpb}.")
        num_blocks = latent_frames // nfpb

        generator = sp.generator
        if generator is None and sp.seed is not None:
            generator = torch.Generator(device=device).manual_seed(sp.seed)

        prompt_embeds = self.encode_prompt(prompt, max_sequence_length=self.transformer.text_len)
        # The text encoder is idle for the rest of the rollout; when neither
        # offload backend manages it, drop it to CPU so its (large, bf16)
        # weights do not co-reside with the transformer + KV cache on-device.
        if not (
            getattr(self.od_config, "enable_cpu_offload", False)
            or getattr(self.od_config, "enable_layerwise_offload", False)
        ):
            self.text_encoder.to("cpu")
            self._empty_device_cache()

        noise = torch.randn(
            [batch_size, self.transformer.in_dim, latent_frames, latent_h, latent_w],
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # KV pool: for local attention, a bounded window; for global attention,
        # only as large as the full clip needs (capped by the upstream default).
        if self.transformer.local_attn_size == -1:
            # Global attention cannot address more than GLOBAL_KV_CACHE_SIZE
            # tokens. Past that the KV window would be silently clamped and the
            # attention slice collapses to width 0 deep in the attention call,
            # so fail here with something actionable instead.
            if latent_frames * frame_seqlen > GLOBAL_KV_CACHE_SIZE:
                max_latents = GLOBAL_KV_CACHE_SIZE // frame_seqlen
                raise ValueError(
                    f"Global attention (local_attn_size=-1) supports at most {max_latents} latent frames "
                    f"at {height}x{width} ({GLOBAL_KV_CACHE_SIZE}-token KV pool / {frame_seqlen} tokens per "
                    f"frame), but this request needs {latent_frames} (num_frames={num_frames}). "
                    f"Set a finite sliding window — local_attn_size <= {max_latents} in the deploy "
                    f"model_config or the model's model_index.json — for constant-memory streaming "
                    f"beyond that length."
                )
            kv_cache_size = min(GLOBAL_KV_CACHE_SIZE, latent_frames * frame_seqlen)
        else:
            kv_cache_size = self.transformer.local_attn_size * frame_seqlen
        kv_cache, crossattn_cache = self._init_caches(batch_size, dtype, device, kv_cache_size)
        denoised_latents: list[torch.Tensor] = []
        # Streaming VAE decode: decode each latent frame the moment it is produced,
        # persisting the decoder's temporal (feat_cache) state across frames — the
        # upstream `cached_decode` contract. Only when returning pixels (not latents).
        #
        # Patch/tile parallel decode runs through the distributed vae.decode() path
        # (which clears the temporal cache and shards spatially across the DiT
        # group). That path is whole-clip, so it is mutually exclusive with the
        # per-latent streaming decode: when it is active, accumulate latents and
        # decode once at the end.
        vae_pp = int(getattr(self.od_config.parallel_config, "vae_patch_parallel_size", 1) or 1)
        stream_decode = output_type != "latent" and vae_pp <= 1
        # Measurement-only: force whole-clip decode (through the CF_VAE_TIMER path) even at
        # vae_pp=1, for a clean single-card decode baseline comparable to the sharded runs.
        import os as _os

        if _os.environ.get("CF_FORCE_WHOLECLIP") == "1" and output_type != "latent":
            stream_decode = False
        decoded_chunks: list[torch.Tensor] = []
        if stream_decode:
            self._decode_stream_begin()

        _cf_stage = _os.environ.get("CF_STAGE_TIMER") == "1"
        _cf_dit_ms = 0.0
        if _cf_stage:
            import time as _time

            torch.xpu.synchronize()
            _cf_dit_t0 = _time.perf_counter()
        current_start_frame = 0
        with self.progress_bar(total=num_blocks) as progress_bar:
            for block_index in range(num_blocks):
                # [B, C, nfpb, H, W] noisy latent block for this chunk.
                noisy_block = noise[:, :, current_start_frame : current_start_frame + nfpb]
                current_start = current_start_frame * frame_seqlen

                denoise_list = (
                    self.denoising_step_list_first_chunk
                    if block_index == 0 and self.denoising_step_list_first_chunk is not None
                    else self.denoising_step_list
                )

                denoised_pred = None
                for step_index, current_timestep in enumerate(denoise_list):
                    # denoise_list is a CPU tensor; use a Python float so the
                    # timestep tensor is created cleanly on the target device.
                    current_timestep = float(current_timestep)
                    timestep = torch.ones([batch_size, nfpb], device=device, dtype=torch.float32) * current_timestep
                    flow_pred = self.transformer(
                        x=noisy_block,
                        timestep=timestep,
                        context=prompt_embeds,
                        kv_cache=kv_cache,
                        current_start=current_start,
                        crossattn_cache=crossattn_cache,
                    )
                    # flow -> x0: [B,C,F,H,W] flattened over (B,F) for per-frame sigma.
                    denoised_pred = (
                        self.scheduler.convert_flow_to_x0(
                            flow_pred=flow_pred.transpose(1, 2).flatten(0, 1),
                            xt=noisy_block.transpose(1, 2).flatten(0, 1),
                            timestep=timestep.flatten(0, 1),
                        )
                        .unflatten(0, (batch_size, nfpb))
                        .transpose(1, 2)
                    )

                    if step_index < len(denoise_list) - 1:
                        # Re-noise x0 toward the next timestep before the next step.
                        next_timestep = float(denoise_list[step_index + 1])
                        x0_flat = denoised_pred.transpose(1, 2).flatten(0, 1)
                        # Use the seeded generator (NOT torch.randn_like, which draws from the
                        # per-rank global RNG). Under tensor-parallelism the transformer is
                        # replicated across ranks, so the re-noise MUST be identical on every
                        # rank; otherwise the first chunk's multi-step denoise diverges per rank
                        # and corrupts the first latent frame (frames 1+ use only the seeded
                        # initial noise and are unaffected).
                        renoise = torch.randn(x0_flat.shape, device=device, dtype=x0_flat.dtype, generator=generator)
                        renoised = self.scheduler.add_noise(
                            x0_flat,
                            renoise,
                            next_timestep * torch.ones([batch_size * nfpb], device=device, dtype=torch.float32),
                        )
                        noisy_block = renoised.unflatten(0, (batch_size, nfpb)).transpose(1, 2)

                denoised_latents.append(denoised_pred)
                # Decode this chunk's latents now (streaming), carrying the VAE
                # temporal cache forward so output is seam-free across chunks.
                if stream_decode:
                    decoded_chunks.append(self._decode_stream_step(denoised_pred))

                # Clean-context refresh: rerun on the clean prediction at
                # context_noise so future blocks attend to clean K/V. Because
                # current_start is unchanged, this overwrites this block's slots.
                context_timestep = (
                    torch.ones([batch_size, nfpb], device=device, dtype=torch.float32) * self.context_noise
                )
                self.transformer(
                    x=denoised_pred,
                    timestep=context_timestep,
                    context=prompt_embeds,
                    kv_cache=kv_cache,
                    current_start=current_start,
                    crossattn_cache=crossattn_cache,
                )

                current_start_frame += nfpb
                progress_bar.update()

        if _cf_stage:
            torch.xpu.synchronize()
            _cf_dit_ms = (_time.perf_counter() - _cf_dit_t0) * 1000.0
        logger.info(
            "[CF_PATH] role=%s output_type=%s stream_decode=%s vae_pp=%s n_latents=%d",
            self._stage_role,
            output_type,
            stream_decode,
            vae_pp,
            len(denoised_latents),
        )
        if output_type == "latent":
            output = torch.cat(denoised_latents, dim=2)  # [B, C, latent_frames, H, W]
            if _cf_stage:
                logger.info("[CF_STAGE] (dit-stage) latents=%d DiT=%.1fms", len(denoised_latents), _cf_dit_ms)
            # Disaggregated DiT stage: also expose the latents on custom_output so
            # the dit2vae bridge reads them from an explicit channel (the default
            # ``.images[0]`` packaging works too, but custom_output survives the
            # ZMQ hop untouched and is unambiguous).
            if self._stage_role == "dit":
                return DiffusionOutput(output=output, custom_output={"latents": output})
        elif stream_decode:
            # Concatenate the per-chunk streaming-decoded video along the time axis.
            output = torch.cat(decoded_chunks, dim=2)  # [B, C, T_video, H, W]
            self._decode_stream_end()
        else:
            # Patch/tile-parallel path: single distributed whole-clip decode.
            video_latents = torch.cat(denoised_latents, dim=2)
            _denormed = self._denorm_latents(video_latents)
            import os as _os

            if _os.environ.get("CF_DUMP_LATENTS"):
                try:
                    import torch as _t

                    _t.save(_denormed.detach().cpu(), _os.environ["CF_DUMP_LATENTS"])
                    logger.info(
                        "[CF_DUMP] saved denormed latents %s to %s",
                        tuple(_denormed.shape),
                        _os.environ["CF_DUMP_LATENTS"],
                    )
                except Exception as _e:
                    logger.warning("[CF_DUMP] failed: %s", _e)
            if _cf_stage:
                torch.xpu.synchronize()
                _cf_vae_t0 = _time.perf_counter()
            output = self.vae.decode(_denormed, return_dict=False)[0]
            if _cf_stage:
                torch.xpu.synchronize()
                _cf_vae_ms = (_time.perf_counter() - _cf_vae_t0) * 1000.0
                logger.info(
                    "[CF_STAGE] vae_pp=%d latents=%d DiT=%.1fms VAE=%.1fms DiT+VAE=%.1fms",
                    vae_pp,
                    len(denoised_latents),
                    _cf_dit_ms,
                    _cf_vae_ms,
                    _cf_dit_ms + _cf_vae_ms,
                )

        return DiffusionOutput(output=output)

    # -----------------------------------------------------------------------
    # VAE stage (disaggregated): decode latents handed over from the DiT stage
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _forward_vae_decode(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """Decode DiT-stage latents to pixels (stage 1 of the disaggregated run).

        The latent tensor rides in the request prompt dict under
        ``extra.latents`` (populated by the ``dit2vae`` stage input processor;
        the ``sampling_params.extra_args`` fallback mirrors GLM-Image's
        ``prior_token_ids`` contract for direct-API callers). Latents are the
        raw model-space output of the DiT rollout, so denorm happens here — the
        same math the aggregated pipeline applies before decode.
        """
        if len(req.prompts) != 1:
            raise ValueError("CausalForcingPipeline (VAE stage) supports a single prompt per request.")
        first_prompt = req.prompts[0]

        sp = req.sampling_params
        output_type = sp.output_type or "np"

        latents = None
        extra_args = getattr(sp, "extra_args", None)
        if isinstance(extra_args, dict):
            latents = extra_args.get("latents")
        if latents is None and isinstance(first_prompt, dict):
            latents = first_prompt.get("extra", {}).get("latents")
        if latents is None:
            # The engine's warmup "dummy run" reaches this stage with no upstream
            # latents. Synthesize a minimal latent so warmup exercises the decode
            # path; real requests always carry latents from dit2vae.
            if req.is_dummy_run():
                height = sp.height or 480
                width = sp.width or 832
                num_frames = sp.num_frames or 81
                latent_frames = (num_frames - 1) // 4 + 1
                in_dim = int(self.vae.config.z_dim) if hasattr(self.vae.config, "z_dim") else 16
                latents = torch.zeros(
                    [1, in_dim, latent_frames, height // 8, width // 8],
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                raise ValueError(
                    "CausalForcingPipeline VAE stage received no latents; expected "
                    "prompt['extra']['latents'] (from dit2vae) or sampling extra_args['latents']."
                )
        if not isinstance(latents, torch.Tensor):
            latents = torch.as_tensor(latents)
        # Latents arrive host-side (CPU) across the stage boundary; place on the
        # VAE device. [B, C, latent_frames, H, W].
        latents = latents.to(device=self.device)

        if output_type == "latent":
            # Pass-through (debug / chaining): hand the latents straight out.
            return DiffusionOutput(output=latents)

        import os as _os

        vae_pp = int(getattr(self.od_config.parallel_config, "vae_patch_parallel_size", 1) or 1)
        stream_decode = vae_pp <= 1
        if _os.environ.get("CF_FORCE_WHOLECLIP") == "1":
            stream_decode = False

        _cf_stage = _os.environ.get("CF_STAGE_TIMER") == "1"
        if _cf_stage:
            import time as _time

            torch.xpu.synchronize()
            _cf_vae_t0 = _time.perf_counter()

        if stream_decode:
            # Per-latent streaming decode, temporal cache persisted across
            # frames — seam-free, identical to the aggregated streaming path.
            self._decode_stream_begin()
            num_latent_frames = latents.shape[2]
            decoded_chunks = [self._decode_stream_step(latents[:, :, i : i + 1]) for i in range(num_latent_frames)]
            output = torch.cat(decoded_chunks, dim=2)
            self._decode_stream_end()
        else:
            # Whole-clip decode (spatial-shard when vae_pp>1); denorm here.
            _denormed = self._denorm_latents(latents)
            output = self.vae.decode(_denormed, return_dict=False)[0]

        if _cf_stage:
            torch.xpu.synchronize()
            _cf_vae_ms = (_time.perf_counter() - _cf_vae_t0) * 1000.0
            logger.info(
                "[CF_STAGE] (vae-stage) vae_pp=%d latent_frames=%d VAE=%.1fms",
                vae_pp,
                latents.shape[2],
                _cf_vae_ms,
            )

        return DiffusionOutput(output=output)

    @staticmethod
    def _empty_device_cache() -> None:
        """Release cached device memory on whichever accelerator is active."""
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()

    # -----------------------------------------------------------------------
    # Streaming VAE decode (per-latent, temporal cache persisted across chunks)
    # -----------------------------------------------------------------------
    # Mirrors the upstream Wan `cached_decode`: the 3D decoder is causal in time
    # and keeps a per-layer `feat_cache` (`_feat_map`). We clear it once at the
    # start of a generation, then feed latents chunk-by-chunk as they are
    # produced — each call carries the temporal state forward, so streaming
    # output is identical (seam-free) to a single whole-clip decode.

    def _denorm_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = self.vae.dtype
        vae_device = next(self.vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=vae_dtype)
        mean = self.vae_latents_mean.to(device=vae_device, dtype=vae_dtype)
        inv_std = self.vae_latents_inv_std.to(device=vae_device, dtype=vae_dtype)
        return latents / inv_std + mean

    def _decode_stream_begin(self) -> None:
        """Reset the VAE temporal cache for a new generation."""
        self.vae.clear_cache()
        self._vae_stream_started = False

    def _decode_stream_step(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode one chunk of latents ([B, C, n, H, W]) with the persisted cache.

        Loops one latent frame at a time (the decoder is per-frame internally),
        keeping ``_feat_map`` alive so the first frame of this chunk attends to
        the temporal context left by the previous chunk. ``first_chunk=True`` only
        on the very first latent frame of the whole generation.
        """
        latents = self._denorm_latents(latents)
        x = self.vae.post_quant_conv(latents)
        num_frame = x.shape[2]
        outs: list[torch.Tensor] = []
        for i in range(num_frame):
            self.vae._conv_idx = [0]
            first = not self._vae_stream_started
            self._vae_stream_started = True
            out = self.vae.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=self.vae._feat_map,
                feat_idx=self.vae._conv_idx,
                first_chunk=first,
            )
            outs.append(out)
        video = torch.cat(outs, dim=2)
        if self.vae.config.patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

            video = unpatchify(video, patch_size=self.vae.config.patch_size)
        return torch.clamp(video, min=-1.0, max=1.0)

    def _decode_stream_end(self) -> None:
        """Clear the VAE temporal cache after a generation completes."""
        self.vae.clear_cache()


class CausalForcingDiTPipeline(CausalForcingPipeline):
    """Stage 0 of the disaggregated Causal-Forcing pipeline (DiT only).

    Builds the text encoder + transformer, runs the framewise rollout, and
    emits raw model-space latents (on ``output`` and ``custom_output['latents']``).
    Registered as a distinct model arch with **no** post-process func, so the
    latent tensor is carried to the VAE stage untouched (a registered
    post-process would convert it to pixels — see ``get_..._post_process_func``).
    """

    _stage_role: ClassVar[str] = "dit"
    _vae_modules: ClassVar[list[str]] = []


class CausalForcingVAEPipeline(CausalForcingPipeline):
    """Stage 1 of the disaggregated Causal-Forcing pipeline (VAE only).

    Builds only the VAE, receives the DiT stage's latents via the ``dit2vae``
    stage input processor, and decodes them to pixels (seam-free streaming
    decode, or spatial-shard whole-clip when ``vae_patch_parallel_size`` > 1).
    Shares the video post-process func with the aggregated pipeline.
    """

    _stage_role: ClassVar[str] = "vae"
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []


def get_causal_forcing_dit_post_process_func(od_config: OmniDiffusionConfig):
    """DiT-stage post-process: route the latent tensor onto the multimodal_output
    ``latent`` channel so it survives the inter-stage connector to the VAE stage.

    Returns a dict (``{"latent": <tensor>}``) rather than the raw tensor: the raw
    tensor would be packaged onto ``.images`` which the connector drops, whereas
    ``multimodal_output`` is preserved (see output_formatter._build_multimodal_output
    and the dynamo stage_worker connector-payload handling). The ``dit2vae`` bridge
    reads ``outputs[0].multimodal_output['latent']``.
    """
    del od_config

    def post_process_func(latents, output_type: str = "latent", sampling_params=None):
        del sampling_params, output_type
        return {"latent": latents, "custom_output": {}}

    return post_process_func


def get_causal_forcing_post_process_func(od_config: OmniDiffusionConfig):
    """Post-process: format decoded [B, C, T, H, W] video via diffusers VideoProcessor."""
    del od_config
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np", sampling_params=None):
        del sampling_params
        if output_type == "latent":
            return video
        return {
            "video": video_processor.postprocess_video(video, output_type=output_type),
            "custom_output": {},
        }

    return post_process_func
