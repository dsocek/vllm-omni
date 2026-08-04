# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CausalForcingPipeline — vllm-omni pipeline for thu-ml/Causal-Forcing.
Adapted from https://github.com/thu-ml/Causal-Forcing (pipeline/causal_inference.py).

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
from collections import deque
from collections.abc import Iterable, Iterator
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

# Latent frames a scene gets when a multi-prompt schedule does not say.
DEFAULT_SCENE_LATENTS = 12

# Temporal RoPE table length built by ``CausalWanModel.__init__`` (rope_params(1024, ...)).
# A rollout may not address a frame position beyond this, which is what makes the
# periodic rebase below necessary for an unbounded stream.
ROPE_TABLE_FRAMES = 1024

# Frame offset at which an open-ended stream rebases its RoPE positions back toward
# zero. Comfortably under ROPE_TABLE_FRAMES so a rebase is never urgent, and high
# enough that rebasing is rare (once per ~768 latent frames, ~3000 video frames).
STREAM_REBASE_AT = 768

# Spread used to derive a scene's seed from the stream seed and the scene's ordinal.
# Any large odd constant works; what matters is that consecutive scenes land far
# apart in the sequence rather than at adjacent offsets.
SCENE_SEED_STRIDE = 0x9E3779B1


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
        # The "full" role builds everything. The disaggregated roles build only the
        # components they actually run.
        self._build_dit = self._stage_role in ("full", "dit")
        self._build_vae = self._stage_role in ("full", "vae")
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)
        # Merge the two sources per key, so a deploy config can override individual
        # constants without having to restate the rest. The engine reads
        # ``model_index.json`` only for ``_class_name`` and never populates
        # ``od_config.model_config`` on the offline path, so without this merge the
        # checkpoint's own constants (notably ``local_attn_size``) would be unreachable.
        model_config = {**_read_model_index(model), **(od_config.model_config or {})}

        # ---- Inference constants (model_index.json / model_config overrides) ----
        self.num_frame_per_block = int(model_config.get("num_frame_per_block", DEFAULT_NUM_FRAME_PER_BLOCK))
        self.context_noise = int(model_config.get("context_noise", DEFAULT_CONTEXT_NOISE))
        timestep_shift = float(model_config.get("timestep_shift", DEFAULT_TIMESTEP_SHIFT))
        denoising_step_list = model_config.get("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)
        denoising_step_list_first_chunk = model_config.get(
            "denoising_step_list_first_chunk", DEFAULT_DENOISING_STEP_LIST_FIRST_CHUNK
        )
        # Sliding-window causal attention. A value of ``-1`` means full global
        # attention, which caps a clip at GLOBAL_KV_CACHE_SIZE tokens (21 latent frames
        # at 480x832). A finite window instead keeps KV memory constant, which is what
        # streaming beyond that cap requires.
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
        # Only prefetch the components this role builds, so that a stage never has to
        # download the other stage's weights.
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
        # The self-attention q/k/v tensors fuse into a single qkv param, so there are
        # always fewer loaded param names than checkpoint tensors. Comparing the two
        # counts would look like a failure, so check coverage against the set of
        # expected destination names instead.
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

    def _build_scene_schedule(self, req: DiffusionRequestBatch, latent_frames: int, nfpb: int) -> list[dict]:
        """Expand the request into a per-block list of scenes to roll out.

        A single-prompt request becomes one scene covering the whole clip, which is
        the original behavior. A request carrying ``extra_args["scenes"]`` becomes a
        sequence of scenes that share one rollout, so the KV history built by scene
        *n* is what scene *n+1* continues from. That shared history is what makes the
        cuts land as camera moves within one take rather than as ten unrelated clips
        pasted together.

        Each scene is ``{"prompt": str, "latents": int, "transition": str}``. Lengths
        are given in latent frames because that is the unit the rollout loop advances
        in, and they get rounded up to a whole number of blocks so a scene never
        changes prompt mid-block.

        ``transition`` picks how much of the previous scene survives into this one:

        ``continue``
            Keep the self-attention history. The new prompt only steers what it can
            from within the established shot, so this reads as a beat change or a
            camera move rather than a new location. Use it for "the same subject now
            does something else". Note that the history dominates the text, so a
            ``continue`` scene asking for a genuinely new setting will mostly be
            ignored.
        ``cut``
            Drop the history and restart positions at zero, so the new scene is a
            fresh shot that happens to share the clip and the decoder's temporal
            state. Use it when the location or subject actually changes.
        """
        sp = req.sampling_params
        raw_scenes = (sp.extra_args or {}).get("scenes")

        if not raw_scenes:
            first = req.prompts[0]
            prompt = first if isinstance(first, str) else (first.get("prompt") or "")
            if not prompt:
                raise ValueError("Prompt is required for Causal-Forcing generation.")
            return [{"prompt": prompt, "latents": latent_frames, "transition": "cut"}]

        scenes: list[dict] = []
        for i, entry in enumerate(raw_scenes):
            if isinstance(entry, str):
                prompt, want, transition = entry, DEFAULT_SCENE_LATENTS, "cut"
            elif isinstance(entry, dict):
                prompt = entry.get("prompt") or ""
                want = int(entry.get("latents") or DEFAULT_SCENE_LATENTS)
                transition = entry.get("transition") or "cut"
            else:
                raise ValueError(f"scenes[{i}] must be a str or a dict, got {type(entry).__name__}.")
            if not prompt:
                raise ValueError(f"scenes[{i}] has no prompt.")
            if transition not in ("continue", "cut"):
                raise ValueError(f"scenes[{i}] transition must be 'continue' or 'cut', got {transition!r}.")
            if want < nfpb:
                raise ValueError(f"scenes[{i}] wants {want} latent frames, which is below one block ({nfpb}).")
            # Round up to a whole number of blocks so the prompt only ever changes
            # on a block boundary.
            blocks = -(-want // nfpb)
            scenes.append({"prompt": prompt, "latents": blocks * nfpb, "transition": transition})

        total = sum(s["latents"] for s in scenes)
        if total != latent_frames:
            raise ValueError(
                f"scenes total {total} latent frames but num_frames={sp.num_frames} needs {latent_frames}. "
                f"Either set num_frames={(total - 1) * 4 + 1} or adjust the per-scene latent counts."
            )
        return scenes

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
    # Open-ended streaming (session-scoped rollout)
    # -----------------------------------------------------------------------

    def open_stream(
        self,
        *,
        height: int = 480,
        width: int = 832,
        seed: int | None = None,
        keep_text_encoder: bool = True,
    ) -> CausalForcingStream:
        """Open a session-scoped rollout that scenes can be appended to over time.

        ``forward`` allocates its KV cache as a local and drops it on return, so
        nothing survives for a later request to continue from. This instead hands
        back a :class:`CausalForcingStream` that owns those caches, so a caller can
        push scenes in as they arrive and keep pulling decoded video out for as long
        as it likes.

        Unlike ``forward`` there is no total length: the stream produces video until
        the caller stops feeding it scenes and closes it. Memory stays flat because
        the KV window is bounded by ``local_attn_size`` and decoded chunks are handed
        to the caller rather than accumulated.

        ``keep_text_encoder`` defaults to True because a stream has to encode prompts
        that do not exist yet, so the encoder cannot be paged out to CPU the way the
        one-shot path does after its up-front encode. Pass False only if the caller
        pre-encodes every prompt itself.
        """
        if self._stage_role == "vae":
            raise RuntimeError("open_stream requires the DiT role; the VAE stage has no rollout to stream.")
        if self.transformer.local_attn_size == -1:
            # Global attention grows its window with the clip and hard-caps at
            # GLOBAL_KV_CACHE_SIZE, so an open-ended stream would eventually die.
            # A finite window is what makes the memory footprint flat.
            raise ValueError(
                "Open-ended streaming requires a finite sliding window; set local_attn_size "
                "(e.g. 21 at 480x832) in the deploy model_config or model_index.json. "
                "local_attn_size=-1 (global attention) cannot stream unbounded length."
            )
        return CausalForcingStream(
            self,
            height=height,
            width=width,
            seed=seed,
            keep_text_encoder=keep_text_encoder,
        )

    def _rollout_block(
        self,
        *,
        noisy_block: torch.Tensor,
        prompt_embeds: torch.Tensor,
        kv_cache: list[dict],
        crossattn_cache: list[dict],
        current_start: int,
        denoise_list: torch.Tensor,
        generator: torch.Generator | None,
        batch_size: int,
        nfpb: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Denoise one latent block and refresh the KV cache with its clean context.

        This is the body of the rollout loop, factored out so the one-shot
        ``forward`` and the open-ended stream run byte-identical math instead of
        two copies that can drift apart.
        """
        denoised_pred = None
        for step_index, current_timestep in enumerate(denoise_list):
            # denoise_list is a CPU tensor; use a Python float so the timestep
            # tensor is created cleanly on the target device.
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
                # Draw from the seeded generator rather than torch.randn_like, which
                # would use the per-rank global RNG. The transformer is replicated
                # across tensor-parallel ranks, so every rank has to produce identical
                # re-noise or the first chunk's multi-step denoise diverges and
                # corrupts the first latent frame.
                renoise = torch.randn(x0_flat.shape, device=device, dtype=x0_flat.dtype, generator=generator)
                renoised = self.scheduler.add_noise(
                    x0_flat,
                    renoise,
                    next_timestep * torch.ones([batch_size * nfpb], device=device, dtype=torch.float32),
                )
                noisy_block = renoised.unflatten(0, (batch_size, nfpb)).transpose(1, 2)

        # Clean-context refresh. Rerun the block on its clean prediction at
        # context_noise so that later blocks attend to clean K/V. Because
        # current_start has not moved, this overwrites the slots this block just
        # wrote rather than appending new ones.
        context_timestep = torch.ones([batch_size, nfpb], device=device, dtype=torch.float32) * self.context_noise
        self.transformer(
            x=denoised_pred,
            timestep=context_timestep,
            context=prompt_embeds,
            kv_cache=kv_cache,
            current_start=current_start,
            crossattn_cache=crossattn_cache,
        )
        return denoised_pred

    # -----------------------------------------------------------------------
    # Forward (framewise rollout)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch, *, on_block=None) -> DiffusionOutput:
        # ``on_block`` (keyword-only) is the DiT block-stream hook: when set, it
        # is called once per rollout block with an intermediate DiffusionOutput
        # (finished=False) carrying that block's raw latents, so the disaggregated
        # DiT stage can stream chunks the moment they are produced. It fires
        # in-worker (see WorkerProc), and the aggregated terminal output is still
        # returned. ``on_block=None`` (the default) is the untouched batch path.
        # VAE stage of the disaggregated pipeline: no rollout, just decode the
        # latents handed over from the DiT stage.
        if self._stage_role == "vae":
            return self._forward_vae_decode(req, on_block=on_block)

        if len(req.prompts) != 1:
            raise ValueError("CausalForcingPipeline supports a single request per batch.")

        sp = req.sampling_params
        height = sp.height or 480
        width = sp.width or 832
        num_frames = sp.num_frames or 81
        output_type = sp.output_type or "np"
        # The DiT stage emits raw model-space latents and the downstream VAE stage owns
        # denormalization and decoding. This role does not even build a VAE, so force
        # the latent path.
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

        # Encode every scene up front, while the text encoder is still resident. The
        # embeddings are small next to the transformer, so holding all of them costs
        # far less than paging the encoder back in at each cut.
        scenes = self._build_scene_schedule(req, latent_frames, nfpb)
        for scene in scenes:
            scene["embeds"] = self.encode_prompt(scene["prompt"], max_sequence_length=self.transformer.text_len)
        if len(scenes) > 1:
            logger.info(
                "[CF_SCENES] %d scenes, latents=%s, transitions=%s",
                len(scenes),
                [s["latents"] for s in scenes],
                [s["transition"] for s in scenes],
            )

        # The text encoder sits idle for the rest of the rollout. When neither offload
        # backend is managing it, move it to CPU so that its weights do not stay
        # resident alongside the transformer and the KV cache.
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
            # Beyond GLOBAL_KV_CACHE_SIZE the KV window gets silently clamped and the
            # attention slice collapses to width zero deep inside the attention call.
            # Fail here instead, where the message can actually explain the fix.
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
        # Streaming decode turns each latent frame into pixels as soon as it is
        # produced, carrying the decoder's temporal feat_cache across frames the way
        # the upstream `cached_decode` does. Patch-parallel decode works on the whole
        # clip at once and clears that cache, so the two modes cannot be combined.
        # When patch-parallel decode is active we accumulate latents and decode once
        # at the end.
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
        # Flatten the scene schedule to one entry per block, so the rollout loop stays a
        # single flat pass and only has to look up "which prompt does this block use".
        block_scene = [i for i, scene in enumerate(scenes) for _ in range(scene["latents"] // nfpb)]

        current_start_frame = 0
        active_scene = -1
        # Token offset of the current scene's first frame. A "cut" restarts attention
        # positions from zero, so the offset the transformer sees is measured from the
        # start of the scene rather than the start of the clip.
        scene_base_frame = 0
        is_scene_opening = False
        with self.progress_bar(total=num_blocks) as progress_bar:
            for block_index in range(num_blocks):
                # [B, C, nfpb, H, W] noisy latent block for this chunk.
                noisy_block = noise[:, :, current_start_frame : current_start_frame + nfpb]

                scene_index = block_scene[block_index]
                is_scene_opening = scene_index != active_scene
                if is_scene_opening:
                    # Crossing into a new scene. The cross-attention cache holds the k/v
                    # of the previous prompt and is only computed on the first block of a
                    # generation, so clearing is_init is what makes it pick up the new
                    # text.
                    for layer_cache in crossattn_cache:
                        layer_cache["is_init"] = False
                    if scenes[scene_index]["transition"] == "cut" and block_index > 0:
                        # Hard cut. Zero the self-attention history and restart positions,
                        # so the new scene is not anchored to the previous location. In
                        # practice the history dominates the text prompt, so without this
                        # a new setting mostly gets ignored and the subject stays put.
                        for layer_cache in kv_cache:
                            layer_cache["global_end_index"].zero_()
                            layer_cache["local_end_index"].zero_()
                        scene_base_frame = current_start_frame
                    active_scene = scene_index
                prompt_embeds = scenes[scene_index]["embeds"]
                current_start = (current_start_frame - scene_base_frame) * frame_seqlen

                # The multi-step first-chunk schedule exists to establish a shot from pure
                # noise with no history to lean on, which is exactly the situation at the
                # start of every cut scene, so reuse it there too.
                denoise_list = (
                    self.denoising_step_list_first_chunk
                    if is_scene_opening
                    and scenes[scene_index]["transition"] == "cut"
                    and self.denoising_step_list_first_chunk is not None
                    else self.denoising_step_list
                )

                denoised_pred = self._rollout_block(
                    noisy_block=noisy_block,
                    prompt_embeds=prompt_embeds,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    denoise_list=denoise_list,
                    generator=generator,
                    batch_size=batch_size,
                    nfpb=nfpb,
                    device=device,
                )

                denoised_latents.append(denoised_pred)
                # Decode this chunk's latents now (streaming), carrying the VAE
                # temporal cache forward so output is seam-free across chunks.
                if stream_decode:
                    decoded_chunks.append(self._decode_stream_step(denoised_pred))

                # DiT block-stream hook: emit this block's latents now.
                # ``_rollout_block`` has already done the clean-context refresh,
                # and that refresh only rewrites *this* block's K/V slots
                # (current_start unchanged), so denoised_pred is final and safe to
                # hand off. The block rides as .output, mirroring the aggregated
                # latent path so the downstream VAE stage reads it the same way.
                # finished=False marks it as non-terminal.
                if on_block is not None:
                    on_block(
                        DiffusionOutput(
                            output=denoised_pred,
                            finished=False,
                            chunk_index=block_index,
                            total_chunks=num_blocks,
                        )
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
            # DiT block-stream lane: every block's latents already went out via
            # on_block (finished=False). The terminal is a bare finished=True
            # sentinel — no payload — so the consumer is not handed the whole
            # clip a second time.
            if on_block is not None and self._stage_role == "dit":
                if _cf_stage:
                    logger.info("[CF_STAGE] (dit-stage, streamed) blocks=%d DiT=%.1fms", num_blocks, _cf_dit_ms)
                return DiffusionOutput(output=None, finished=True)
            output = torch.cat(denoised_latents, dim=2)  # [B, C, latent_frames, H, W]
            if _cf_stage:
                logger.info("[CF_STAGE] (dit-stage) latents=%d DiT=%.1fms", len(denoised_latents), _cf_dit_ms)
            # No early return for the disaggregated DiT stage: the shared
            # ``return DiffusionOutput(output=output)`` below already returns the
            # latents as the single output channel, which is what the dit2vae
            # bridge reads (``multimodal_output['latent']``, falling back to
            # ``custom_output['latents']`` / ``.images[0]``). One exit point.
        elif stream_decode:
            # Concatenate the per-chunk streaming-decoded video along the time axis.
            output = self._concat_stream_chunks(decoded_chunks)  # [B, C, T_video, H, W]
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
    def _forward_vae_decode(self, req: DiffusionRequestBatch, *, on_block=None) -> DiffusionOutput:
        """Decode DiT-stage latents to pixels (stage 1 of the disaggregated run).

        The latent tensor rides in the request prompt dict under
        ``extra.latents`` (populated by the ``dit2vae`` stage input processor;
        the ``sampling_params.extra_args`` fallback mirrors GLM-Image's
        ``prior_token_ids`` contract for direct-API callers). Latents are the
        raw model-space output of the DiT rollout, so denorm happens here — the
        same math the aggregated pipeline applies before decode.

        ``on_block`` (keyword-only) is the VAE pixel-stream hook, symmetric to
        the DiT block-stream lane: when set on the single-card streaming path, it
        fires once per decoded latent frame with an intermediate DiffusionOutput
        (finished=False) carrying that frame's pixels, so the router can push CMAF
        segments live. The temporal ``feat_cache`` is persisted across frames
        exactly as the aggregated stream does, so the emitted pixels are seam-free
        and byte-identical to the whole-clip decode. ``on_block=None`` is the
        untouched aggregated path.
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
            # The engine's warmup dummy run reaches this stage with no upstream
            # latents, so synthesize a minimal one to keep warmup exercising the
            # decode path. Real requests always carry latents from dit2vae.
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
        # Spatial sharding divides each frame, so it composes with a per-frame
        # streaming decode; tiling divides the clip and does not. Streaming is what
        # the pipelined router needs, and it is also the faster of the two here, so
        # prefer it wherever the split is spatial.
        stream_decode = vae_pp <= 1 or self._stream_shard_split_dim() is not None
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
            decoded_chunks = []
            for i in range(num_latent_frames):
                pixels = self._decode_stream_step(latents[:, :, i : i + 1])
                decoded_chunks.append(pixels)
                # VAE pixel-stream hook: emit this frame's pixels the moment they
                # are decoded. The feat_cache carried by _decode_stream_step keeps
                # the stream seam-free, so a streamed frame is byte-identical to
                # its slice of the whole-clip output. finished=False marks it
                # non-terminal (mirrors the DiT on_block lane).
                #
                # ndim == 5 is the same guard _concat_stream_chunks applies: a
                # sharded decode leaves an empty placeholder on every rank but
                # rank 0, and streaming a placeholder out would put a frame-shaped
                # hole on the wire rather than skipping it.
                if on_block is not None and pixels.ndim == 5:
                    on_block(
                        DiffusionOutput(
                            output=pixels,
                            finished=False,
                            chunk_index=i,
                            total_chunks=num_latent_frames,
                        )
                    )
            output = self._concat_stream_chunks(decoded_chunks)
            self._decode_stream_end()
            # Streamed lane: every frame already went out via on_block. Return a
            # bare terminal sentinel (no payload) so the consumer is not handed
            # the whole clip a second time — same contract as the DiT lane.
            if on_block is not None:
                return DiffusionOutput(output=None, finished=True)
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
    # This mirrors the upstream Wan `cached_decode`. The 3D decoder is causal in time
    # and keeps a per-layer `feat_cache`, which we clear once at the start of a
    # generation and then feed chunk by chunk as latents are produced. Each call
    # carries the temporal state forward, so the streamed output is seam-free and
    # identical to decoding the whole clip in one pass.

    def _denorm_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = self.vae.dtype
        vae_device = next(self.vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=vae_dtype)
        mean = self.vae_latents_mean.to(device=vae_device, dtype=vae_dtype)
        inv_std = self.vae_latents_inv_std.to(device=vae_device, dtype=vae_dtype)
        return latents / inv_std + mean

    def _stream_shard_split_dim(self) -> str | None:
        """``"height"``/``"width"`` if this stream can decode sharded, else ``None``.

        A probe: it installs nothing, so it is safe to call while still deciding
        whether to stream at all. ``tile`` mode answers ``None`` — tiles are a
        whole-clip split and there is nothing to shard within one latent frame.
        """
        ready = getattr(self.vae, "spatial_shard_decode_split_dim_if_ready", None)
        return ready() if ready is not None else None

    def _decode_stream_begin(self) -> None:
        """Reset the VAE temporal cache for a new generation.

        This is also where a sharded stream is committed to, once, for its whole
        life. Two reasons it cannot be decided per chunk: installing the patch
        mutates the decoder module, and every rank has to agree for every chunk —
        a rank that skipped one would leave the others waiting forever inside a
        halo exchange, which reads as a hang rather than an error.
        """
        self.vae.clear_cache()
        self._vae_stream_started = False
        split_dim = self._stream_shard_split_dim()
        if split_dim is not None:
            from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard

            wan_spatial_shard.install_wan_spatial_shard_decode(
                self.vae,
                self.vae.distributed_executor.group,
                split_dim=split_dim,
            )
        self._vae_stream_split_dim = split_dim

    # Decorated in its own right, not just via a caller: every other decode entry
    # point (``forward``, ``_forward_vae_decode``, ``blocks``) carries @torch.no_grad,
    # so for a long time this inherited one. The disaggregated VAE stage calls it
    # directly from ``session_decode_step``, which the engine runs as a control RPC
    # between scheduler steps -- outside any inference context. The weights are
    # inference tensors, so autograd then refuses to save them for backward and the
    # decode dies with "Inference tensors cannot be saved for backward".
    @torch.no_grad()
    def _decode_stream_step(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode one chunk of latents ([B, C, n, H, W]) with the persisted cache.

        Loops one latent frame at a time (the decoder is per-frame internally),
        keeping ``_feat_map`` alive so the first frame of this chunk attends to
        the temporal context left by the previous chunk. ``first_chunk=True`` only
        on the very first latent frame of the whole generation.

        Under a sharded stream the patched ``decoder.forward`` splits each frame
        across ranks, exchanges halos, and gathers on rank 0 — so the temporal chain
        stays intact per rank while the spatial work is divided. That is the only
        way to split this decode: two ranks taking alternate *chunks* would each
        decode against a cache that never saw the other's frames, which produces a
        seam at every boundary rather than an error. Ranks other than 0 run every
        collective and return an empty placeholder, mirroring the whole-clip
        sharded path's ``broadcast_result=False`` contract.
        """
        latents = self._denorm_latents(latents)
        produce_output = not self._sharded_stream_bystander()
        with self.vae._execution_context():
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
                if produce_output:
                    outs.append(out)
            if not produce_output:
                # An empty tensor rather than this rank's shard: a shard is a stripe
                # of an image, and handing one back as if it were a frame is the kind
                # of thing that survives all the way to a video file.
                return latents.new_zeros(0)
            video = torch.cat(outs, dim=2)
        if self.vae.config.patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

            video = unpatchify(video, patch_size=self.vae.config.patch_size)
        return torch.clamp(video, min=-1.0, max=1.0)

    def _sharded_stream_bystander(self) -> bool:
        """True when this rank takes part in a sharded decode but assembles nothing."""
        if getattr(self, "_vae_stream_split_dim", None) is None:
            return False
        import torch.distributed as dist

        group = self.vae.distributed_executor.group
        return dist.get_world_size(group) > 1 and dist.get_rank(group) != 0

    @staticmethod
    def _concat_stream_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
        """Join per-chunk decoded video along time.

        Tolerates the empty placeholders a sharded decode leaves on every rank but
        rank 0, which would otherwise fail the concat on a dimension they do not have.
        """
        frames = [chunk for chunk in chunks if chunk.ndim == 5]
        if not frames:
            return chunks[0] if chunks else torch.zeros(0)
        return torch.cat(frames, dim=2)

    def _decode_stream_end(self) -> None:
        """Clear the VAE temporal cache after a generation completes."""
        self.vae.clear_cache()


class CausalForcingStream:
    """A rollout that outlives a single request, so scenes can be appended over time.

    ``CausalForcingPipeline.forward`` needs its whole scene schedule up front: it
    allocates the KV cache as a local, sizes the noise tensor to the total length,
    and drops both on return. That makes an open-ended stream impossible — request
    *n+1* has nothing to continue from, no matter where it is routed.

    This class holds the state that has to survive instead:

    - the self-attention KV window and the cross-attention cache,
    - the VAE decoder's temporal ``feat_cache``, so chunk boundaries stay seam-free,
    - the frame cursor and RoPE base that place the next block in time.

    Memory is flat rather than proportional to what has been produced. The KV window
    is bounded by ``local_attn_size``, noise is drawn one block at a time, and decoded
    chunks are handed to the caller instead of accumulated. A stream can therefore run
    until the caller stops it.

    Usage is push-then-drain, which is what a session-oriented service does when a
    prompt arrives::

        stream = pipe.open_stream(seed=42)
        stream.push_scene("a reef wall in clear water", latents=16)
        for chunk in stream.blocks():
            ...                       # [B, C, t, H, W] pixels, ready to encode
        stream.push_scene("a turtle over sand", latents=16, transition="cut")
        for chunk in stream.blocks():
            ...
        stream.close()

    ``blocks()`` returns when the queue runs dry rather than ending the stream, so it
    can be re-entered after each push for as long as the caller keeps supplying work.

    Seeds work at two levels. ``open_stream(seed=...)`` seeds the whole rollout, so a
    storyboard replays identically end to end. ``push_scene(seed=...)`` overrides one
    scene, which is how a single shot gets re-rolled without touching the others.

    That independence is why every scene gets its *own* generator, seeded from the
    stream seed and the scene's ordinal, rather than all of them sharing one. Sharing
    would make each scene's noise depend on how many draws preceded it, so overriding
    one scene's seed would shift every later scene's position in the shared sequence
    and silently re-render the rest of the board. Deriving per scene means scene *n*'s
    noise is a function of ``(stream_seed, n)`` alone, which is what makes re-rolling
    a single shot a local edit.
    """

    def __init__(
        self,
        pipe: CausalForcingPipeline,
        *,
        height: int = 480,
        width: int = 832,
        seed: int | None = None,
        keep_text_encoder: bool = True,
    ) -> None:
        self.pipe = pipe
        self.height = height
        self.width = width
        self.keep_text_encoder = keep_text_encoder

        self.device = pipe.device
        self.dtype = pipe.dtype
        self.batch_size = 1
        self.nfpb = pipe.num_frame_per_block

        tf = pipe.transformer
        self.latent_h = height // 8
        self.latent_w = width // 8
        self.frame_seqlen = (self.latent_h // tf.patch_size[1]) * (self.latent_w // tf.patch_size[2])

        # Kept as the seed rather than a live generator: each scene derives its own
        # from this plus its ordinal, so no shared draw position exists to disturb.
        self.seed = seed

        # Bounded window, so this allocation is the stream's whole steady-state KV
        # cost regardless of how long it runs. open_stream rejects local_attn_size=-1.
        kv_cache_size = tf.local_attn_size * self.frame_seqlen
        self.kv_cache, self.crossattn_cache = pipe._init_caches(self.batch_size, self.dtype, self.device, kv_cache_size)

        # Where the next block lands, and the frame the current scene's positions are
        # measured from. A cut rebases the latter to "now"; a long unbroken run of
        # continues rebases it via _rebase_positions before the RoPE table runs out.
        self.current_start_frame = 0
        self.scene_base_frame = 0

        self._pending: deque[dict] = deque()
        # Monotonic across the stream's life, not a queue index: it has to keep
        # counting scenes that have already been drained, or a scene's derived seed
        # would depend on when the caller happened to drain.
        self._scenes_queued = 0
        self._closed = False
        self._first_block = True
        self.latents_emitted = 0
        self.scenes_played = 0

        self._decodes = pipe.vae is not None
        if self._decodes:
            pipe._decode_stream_begin()

    # -- scene queue --------------------------------------------------------

    def push_scene(
        self,
        prompt: str,
        *,
        latents: int = DEFAULT_SCENE_LATENTS,
        transition: str = "continue",
        seed: int | None = None,
    ) -> None:
        """Queue a scene. Encoding happens here so ``blocks()`` never stalls on text.

        ``transition`` defaults to ``"continue"`` here, the opposite of the one-shot
        scene schedule's default. A stream is a single ongoing shot that a caller is
        steering, so keeping the history is the common case; ``"cut"`` is the explicit
        request for a new location.

        ``seed`` overrides this scene's derived seed, so one shot can be re-rolled
        without disturbing the others -- scenes do not share a draw sequence, so an
        override is genuinely local. It fixes the scene's *noise*, which is the whole
        story for a ``cut``: that zeroes the attention history, so the same seed and
        prompt reproduce the scene exactly. A ``continue`` also attends to the
        preceding scenes' KV window, so it reproduces exactly only when what came
        before is unchanged -- which is the case when replaying a fixed storyboard, and
        is not when you have edited an earlier scene.
        """
        if self._closed:
            raise RuntimeError("Cannot push a scene onto a closed stream.")
        if not prompt or not prompt.strip():
            raise ValueError("Scene prompt must be non-empty.")
        if transition not in ("continue", "cut"):
            raise ValueError(f"transition must be 'continue' or 'cut', got {transition!r}.")
        if latents < self.nfpb:
            raise ValueError(f"latents={latents} is below one block ({self.nfpb}).")
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be an int or None, got {type(seed).__name__}.")

        if self.keep_text_encoder and next(self.pipe.text_encoder.parameters()).device != self.device:
            # forward() parks the encoder on CPU after its up-front encode. A stream
            # encodes prompts that do not exist yet, so bring it back.
            self.pipe.text_encoder.to(self.device)

        blocks = -(-latents // self.nfpb)  # round up: prompts change on block boundaries
        embeds = self.pipe.encode_prompt(prompt, max_sequence_length=self.pipe.transformer.text_len)
        scene_seed = seed if seed is not None else self._derive_seed(self._scenes_queued)
        self._pending.append(
            {
                "prompt": prompt,
                "latents": blocks * self.nfpb,
                "transition": transition,
                "seed": scene_seed,
                "embeds": embeds,
                # Resume state, so a caller that stops mid-scene can come back to it.
                # ``remaining`` is the cursor; ``started`` guards the once-per-scene
                # cache transitions; ``generator`` has to be the *same* generator
                # across re-entries, because a fresh one reseeded to scene_seed would
                # redraw the noise this scene's earlier blocks already used and repeat
                # them. See blocks() for why any of this is reachable.
                "remaining": blocks,
                "started": False,
                "generator": (
                    torch.Generator(device=self.device).manual_seed(scene_seed) if scene_seed is not None else None
                ),
            }
        )
        self._scenes_queued += 1

    def _derive_seed(self, ordinal: int) -> int | None:
        """Scene ``ordinal``'s seed, from the stream seed. None when unseeded.

        Returning None keeps an unseeded stream genuinely random: there is no seed to
        derive from, so the scene falls through to torch's default generator.
        """
        if self.seed is None:
            return None
        # Masked to 63 bits: manual_seed takes an int64, and the product would
        # otherwise overflow it for a large stream seed.
        return (self.seed + ordinal * SCENE_SEED_STRIDE) & ((1 << 63) - 1)

    def close(self) -> None:
        """Stop accepting scenes and release the decoder's temporal cache."""
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        if self._decodes:
            self.pipe._decode_stream_end()

    @property
    def pending_latents(self) -> int:
        """Latent frames still queued — what a service would use for backpressure.

        Counts each scene's *undrained* blocks, so a scene that is half way through
        reports what is left rather than what it started with. Summing ``latents``
        would make a stream draining one block at a time look permanently full.
        """
        return sum(s["remaining"] * self.nfpb for s in self._pending)

    # -- RoPE rebase --------------------------------------------------------

    def _rebase_positions(self, delta_frames: int) -> None:
        """Shift every cached position ``delta_frames`` earlier, losing no history.

        The RoPE table is finite (``ROPE_TABLE_FRAMES``), so an unbroken run of
        ``continue`` scenes would eventually index past its end. Cuts rebase for free
        by discarding history, but a stream that never cuts still has to stay in range.

        Keys are RoPE-applied *before* being written to the cache, so absolute
        position is baked into them and the cursor cannot simply be moved. What makes
        this exact is that the rotation between frame ``f`` and ``f - delta`` is
        ``conj(freqs[delta])`` for every ``f`` — the same factor everywhere — so one
        elementwise multiply over the temporal RoPE channels rebases the entire cache
        while leaving all query/key relative geometry untouched.
        """
        tf = self.pipe.transformer
        freqs_t = tf.freqs[0]
        if freqs_t.device != self.device:
            freqs_t = freqs_t.to(self.device)
            tf.freqs = [f.to(self.device) for f in tf.freqs]
        # exp(-i*delta*theta): the phase to undo delta frames of temporal rotation.
        shift = torch.conj(freqs_t[delta_frames])
        n_temporal = shift.shape[0]

        for layer_cache in self.kv_cache:
            k = layer_cache["k"]
            # [B, size, heads, head_dim] real -> complex pairs. Temporal channels come
            # first in the concatenated (temporal, height, width) RoPE layout.
            k_c = torch.view_as_complex(k.float().unflatten(-1, (-1, 2)))
            k_c[..., :n_temporal] = k_c[..., :n_temporal] * shift
            k.copy_(torch.view_as_real(k_c).flatten(-2).to(k.dtype))
            # global_end_index is absolute-token bookkeeping and has to move with the
            # positions, or the next block's slot arithmetic jumps by delta.
            layer_cache["global_end_index"] -= delta_frames * self.frame_seqlen

        self.scene_base_frame += delta_frames
        logger.info(
            "[CF_STREAM] rebased RoPE positions by %d frames (cursor now %d)",
            delta_frames,
            self.current_start_frame - self.scene_base_frame,
        )

    # -- rollout ------------------------------------------------------------

    @torch.no_grad()
    def blocks(self) -> Iterator[torch.Tensor]:
        """Roll out every queued scene, yielding one decoded chunk per block.

        Returns when the queue runs dry — not when the stream ends — so the caller can
        push more scenes and re-enter. Yields ``[B, C, t, H, W]`` pixels, or raw
        latents when the pipeline has no VAE (the disaggregated DiT role).

        **Abandoning this generator part-way through a scene is supported**, and the
        pipelined service does exactly that: it drains with ``max_blocks=1`` and breaks,
        which drops the generator after one block and builds a fresh one for the next
        call. So per-scene progress lives on the queued scene rather than in this
        frame's locals, and the scene is only removed once its last block is out. An
        earlier version popped the scene on entry, which meant a one-block drain took
        the scene away with the discarded generator: a 25-latent scene emitted its first
        block and then reported an empty queue, dropping the other 24 with nothing
        logged as wrong. The aggregated path never saw it because it drains with
        ``max_blocks=None`` inside a single generator's lifetime.
        """
        if self._closed:
            raise RuntimeError("Cannot roll out a closed stream.")
        pipe = self.pipe
        tf = pipe.transformer

        while self._pending:
            scene = self._pending[0]

            # The cache transitions below are per *scene*, not per generator, so they
            # must not run again when a partly-drained scene is resumed -- a second cut
            # would re-zero a history the scene's own earlier blocks are attending to.
            if not scene["started"]:
                scene["started"] = True
                n_blocks = scene["latents"] // self.nfpb

                # A cut drops the history so the new prompt is not anchored to the old
                # location; a continue keeps it. Either way the cross-attention cache is
                # holding the previous prompt's k/v and is only recomputed when is_init is
                # cleared, so clearing it is what makes the new text take effect at all.
                for layer_cache in self.crossattn_cache:
                    layer_cache["is_init"] = False
                if scene["transition"] == "cut" and not self._first_block:
                    for layer_cache in self.kv_cache:
                        layer_cache["global_end_index"].zero_()
                        layer_cache["local_end_index"].zero_()
                    self.scene_base_frame = self.current_start_frame
                elif not self._first_block:
                    # Continues accumulate position without bound. Rebase before the
                    # cursor can reach the end of the RoPE table.
                    cursor = self.current_start_frame - self.scene_base_frame
                    if cursor + n_blocks * self.nfpb >= STREAM_REBASE_AT:
                        self._rebase_positions(cursor)

                # A cut starts from pure noise with no usable history, which is what the
                # multi-step first-chunk schedule is for. Reuse it at every cut, and at
                # the stream's very first block.
                scene["opens_shot"] = scene["transition"] == "cut" or self._first_block

            # Only the scene's *first* block opens a shot; the rest continue it. Held on
            # the scene so a resumed drain does not re-apply the first-chunk schedule to
            # a middle block, which would denoise it as if it had no history.
            denoise_list = (
                pipe.denoising_step_list_first_chunk
                if scene["opens_shot"] and pipe.denoising_step_list_first_chunk is not None
                else pipe.denoising_step_list
            )

            # Each scene's own generator, from its explicit seed or one derived at push
            # time. Sharing one across scenes would make each scene's noise depend on the
            # number of draws before it, so overriding one seed would shift every later
            # scene and silently re-render the board. Created at push time rather than
            # here so its draw position survives a mid-scene re-entry.
            gen = scene["generator"]

            while scene["remaining"] > 0:
                # Draw noise per block. forward() sizes one tensor for the whole clip,
                # which is exactly what an unbounded stream must not do.
                noisy_block = torch.randn(
                    [self.batch_size, tf.in_dim, self.nfpb, self.latent_h, self.latent_w],
                    device=self.device,
                    dtype=self.dtype,
                    generator=gen,
                )
                denoised = pipe._rollout_block(
                    noisy_block=noisy_block,
                    prompt_embeds=scene["embeds"],
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(self.current_start_frame - self.scene_base_frame) * self.frame_seqlen,
                    denoise_list=denoise_list,
                    generator=gen,
                    batch_size=self.batch_size,
                    nfpb=self.nfpb,
                    device=self.device,
                )
                self.current_start_frame += self.nfpb
                self.latents_emitted += self.nfpb
                self._first_block = False
                # Only the first block of a cut scene opens a shot; the rest of the
                # scene continues it. Cleared on the scene, so a drain that resumes
                # here also sees the shot as already open.
                denoise_list = pipe.denoising_step_list
                scene["opens_shot"] = False

                # Both before the yield: control may never come back. A caller that
                # breaks out of this generator has still consumed the block, and this is
                # the state that tells the next drain so.
                scene["remaining"] -= 1
                if scene["remaining"] == 0:
                    self._pending.popleft()
                    self.scenes_played += 1

                # Hand the chunk straight out. Accumulating here is what makes the
                # one-shot path's memory grow with length.
                yield pipe._decode_stream_step(denoised) if self._decodes else denoised


class CausalForcingDiTPipeline(CausalForcingPipeline):
    """Stage 0 of the disaggregated Causal-Forcing pipeline (DiT only).

    Builds the text encoder + transformer, runs the framewise rollout, and
    emits raw model-space latents on ``output``. Registered as a distinct model
    arch so it gets its own post-process func
    (``get_causal_forcing_dit_post_process_func``), which routes the latent onto
    ``multimodal_output['latent']`` (the channel the inter-stage connector
    preserves) instead of decoding it to pixels the way the aggregated
    pipeline's post-process would.
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
