# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CausalWanModel — causal Wan2.1 DiT with a dict-based KV cache.
Adapted from https://github.com/thu-ml/Causal-Forcing (wan/modules/causal_model.py,
pipeline/causal_inference.py).

Port of thu-ml/Causal-Forcing (``causal-forcing++/framewise-1step``), a
Wan2.1-T2V-1.3B-based causal / autoregressive few-step video diffusion model.
Video latents are produced block-by-block (``num_frame_per_block``) with a KV
cache so new frames attend causally to already-generated history.

This is the T2V-only cousin of the DreamZero ``CausalWanModel``: the robotics
conditioning (action/state encoders, image encoder, action/state RoPE tables)
and the i2v branch are removed. Unlike DreamZero's paged AR-Diffusion cache,
this model uses the upstream repo's dict-based KV cache
(``{k, v, global_end_index, local_end_index}`` per layer) so the clean-context
refresh pass (rerun at ``context_noise``) can OVERWRITE the current block's K/V
slots in place — the mechanism few-step Causal Forcing relies on.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_omni.diffusion.attention.layer import Attention

# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional embedding for timesteps."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}.")
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, dtype=position.dtype, device=position.device).div(half)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len: int, dim: int) -> torch.Tensor:
    """Precompute complex-valued RoPE frequencies (polar form).
    Returns: complex tensor [max_seq_len, dim // 2]
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}.")
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(10000, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x using precomputed complex freqs.

    x: [B, seq_len, n_heads, head_dim]; freqs: [seq_len, 1, head_dim // 2].
    """
    B, seq_len, n, _ = x.shape
    x = torch.view_as_complex(x.to(torch.float64).reshape(B, seq_len, n, -1, 2))
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm wrapper used by the Wan blocks (affine off by default)."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)


class DistributedRMSNorm(nn.Module):
    """RMSNorm that computes global RMS across tensor-parallel ranks.

    At tp_size == 1 this is a plain RMSNorm over the last dim; the upstream
    model uses ``WanRMSNorm(dim)`` on the full q/k vector before the head split,
    which is numerically identical.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if param.shape == loaded_weight.shape:
            param.data.copy_(loaded_weight)
            return

        tp_size = get_tensor_model_parallel_world_size()
        if loaded_weight.shape[0] % tp_size != 0:
            raise ValueError(
                f"Cannot shard RMSNorm weight of shape {tuple(loaded_weight.shape)} across tp_size={tp_size}."
            )
        shard_size = loaded_weight.shape[0] // tp_size
        start_idx = get_tensor_model_parallel_rank() * shard_size
        shard = loaded_weight.narrow(0, start_idx, shard_size)
        if param.shape != shard.shape:
            raise ValueError(f"RMSNorm shard shape mismatch: param={tuple(param.shape)}, shard={tuple(shard.shape)}.")
        param.data.copy_(shard)

    def _local_sum_sq(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        x_float = x.float()
        local_sum_sq = x_float.pow(2).sum(dim=-1, keepdim=True)
        return x_float, local_sum_sq, x.shape[-1]

    def _scale(
        self,
        x_float: torch.Tensor,
        global_sum_sq: torch.Tensor,
        global_count: int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        mean_sq = global_sum_sq / global_count
        return (x_float * torch.rsqrt(mean_sq + self.eps)).type_as(x) * self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        x_float, local_sum_sq, local_count = self._local_sum_sq(x)
        if tp_size > 1:
            global_sum_sq = tensor_model_parallel_all_reduce(local_sum_sq)
            global_count = local_count * tp_size
        else:
            global_sum_sq = local_sum_sq
            global_count = local_count
        return self._scale(x_float, global_sum_sq, global_count, x)


def fused_qk_rms_norm(
    norm_q: nn.Module,
    norm_k: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply q/k :class:`DistributedRMSNorm` with a single fused TP all-reduce.

    Numerically identical to ``norm_q(q), norm_k(k)`` (all-reduce is elementwise).
    Falls back to independent application when either norm is not a
    DistributedRMSNorm (e.g. nn.Identity when qk_norm=False).
    """
    if not (isinstance(norm_q, DistributedRMSNorm) and isinstance(norm_k, DistributedRMSNorm)):
        return norm_q(q), norm_k(k)

    assert q.shape == k.shape, "fused_qk_rms_norm requires q and k to have the same shape."
    tp_size = get_tensor_model_parallel_world_size()
    q_float, q_sum_sq, count = norm_q._local_sum_sq(q)
    k_float, k_sum_sq, _ = norm_k._local_sum_sq(k)
    if tp_size > 1:
        packed = torch.cat([q_sum_sq, k_sum_sq], dim=-1)
        packed = tensor_model_parallel_all_reduce(packed)
        q_sum_sq, k_sum_sq = packed[..., 0:1], packed[..., 1:2]
        count = count * tp_size
    q_out = norm_q._scale(q_float, q_sum_sq, count, q)
    k_out = norm_k._scale(k_float, k_sum_sq, count, k)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Cross-Attention (text-to-video)
# ---------------------------------------------------------------------------


class WanT2VCrossAttention(nn.Module):
    """Text-to-video cross-attention with a session-static k/v cache.

    ``context`` (the text embedding) is constant within a generation, so k/v are
    computed once (on ``is_init``) and reused across every block/step.
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")
        self.tp_num_heads = num_heads // tp_size
        self.tp_inner_dim = self.tp_num_heads * self.head_dim
        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False)
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = Attention(
            self.tp_num_heads,
            self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        n, d = self.tp_num_heads, self.head_dim
        q = self.norm_q(self.q(x)).unflatten(2, (n, d))
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).unflatten(2, (n, d))
                v = self.v(context).unflatten(2, (n, d))
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).unflatten(2, (n, d))
            v = self.v(context).unflatten(2, (n, d))
        x = self.attn(q, k, v)
        x = x.flatten(2)
        x = self.o(x)
        return x


# ---------------------------------------------------------------------------
# Self-Attention with causal masking + dict KV cache
# ---------------------------------------------------------------------------


class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with an in-place dict KV cache.

    The cache is a mutable dict ``{"k", "v", "global_end_index", "local_end_index"}``
    (allocated by the pipeline). New K/V are written at token positions derived
    from ``current_start``; because ``current_start`` is held fixed across a
    block's denoise steps AND its clean-context refresh pass, later writes
    overwrite the same slots — the refresh pass replaces noisy K/V with clean.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")
        self.tp_num_heads = num_heads // tp_size
        self.tp_inner_dim = self.tp_num_heads * self.head_dim
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        # Fused QKV projection: q/k/v all come from x. No GQA (kv_heads == heads).
        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            bias=True,
        )
        if self.qkv.total_num_kv_heads != self.qkv.total_num_heads:
            raise ValueError("Self-attn QKV fusion requires no GQA (total_num_kv_heads == total_num_heads).")
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False)
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = Attention(
            self.tp_num_heads,
            self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        frame_seqlen: int,
        kv_cache: dict,
        current_start: int = 0,
    ) -> torch.Tensor:
        """Inference-only forward (dict KV-cache path)."""
        n, d = self.tp_num_heads, self.head_dim

        qkv, _ = self.qkv(x)
        qk_size = self.tp_num_heads * self.head_dim
        q, k, v = qkv.split([qk_size, qk_size, qk_size], dim=-1)
        q, k = fused_qk_rms_norm(self.norm_q, self.norm_k, q, k)
        q = q.unflatten(2, (n, d))
        k = k.unflatten(2, (n, d))
        v = v.unflatten(2, (n, d))

        roped_query = rope_apply(q, freqs).type_as(v)
        roped_key = rope_apply(k, freqs).type_as(v)

        # Global attention window when local_attn_size == -1 is the full cache;
        # otherwise a temporal window of local_attn_size latent frames.
        kv_cache_size = kv_cache["k"].shape[1]
        max_attention_size = kv_cache_size if self.local_attn_size == -1 else self.local_attn_size * frame_seqlen

        current_end = current_start + roped_query.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        num_new_tokens = roped_query.shape[1]
        global_end = kv_cache["global_end_index"].item()
        local_end = kv_cache["local_end_index"].item()

        if self.local_attn_size != -1 and current_end > global_end and (num_new_tokens + local_end > kv_cache_size):
            # Roll the cache left, preserving the first ``sink_tokens`` frames.
            num_evicted_tokens = num_new_tokens + local_end - kv_cache_size
            num_rolled_tokens = local_end - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
                :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
            ].clone()
            kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
                :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
            ].clone()
            local_end_index = local_end + current_end - global_end - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            # Write new K/V directly up to current_end. When current_start is
            # unchanged (repeated denoise / refresh of the same block), this
            # overwrites the same slots in place.
            local_end_index = local_end + current_end - global_end
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v

        window_start = max(0, local_end_index - max_attention_size)
        x = self.attn(
            roped_query,
            kv_cache["k"][:, window_start:local_end_index],
            kv_cache["v"][:, window_start:local_end_index],
        )
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        x = x.flatten(2)
        x = self.o(x)
        return x


# ---------------------------------------------------------------------------
# Attention Block
# ---------------------------------------------------------------------------


class CausalWanAttentionBlock(nn.Module):
    """Transformer block: self-attn + cross-attn + FFN with 6-param modulation."""

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if cross_attn_type != "t2v_cross_attn":
            raise ValueError(f"Only t2v_cross_attn is supported, got {cross_attn_type!r}.")
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            ColumnParallelLinear(dim, ffn_dim, bias=True, gather_output=False, return_bias=False),
            nn.GELU(approximate="tanh"),
            RowParallelLinear(ffn_dim, dim, bias=True, input_is_parallel=True, return_bias=False),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        frame_seqlen: int,
        context: torch.Tensor,
        kv_cache: dict,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
    ) -> torch.Tensor:
        # e: [B, L, 6, dim] (per-token modulation)
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn(
            x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
            freqs=freqs,
            frame_seqlen=frame_seqlen,
            kv_cache=kv_cache,
            current_start=current_start,
        )
        x = x + (y * e[2].squeeze(2))

        x = x + self.cross_attn(self.norm3(x), context, crossattn_cache=crossattn_cache)
        y = self.ffn(self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        x = x + (y * e[5].squeeze(2))
        return x


# ---------------------------------------------------------------------------
# Output Head
# ---------------------------------------------------------------------------


class CausalHead(nn.Module):
    """Output norm + linear with 2-param modulation."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        out_channels = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_channels)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # e: [B, L, 1, dim] (per-token time embedding)
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class CausalWanModel(nn.Module):
    """Causal Wan2.1 video diffusion transformer (T2V, few-step, KV-cached).

    Default geometry matches Wan2.1-T2V-1.3B (dim=1536, ffn_dim=8960,
    num_layers=30, num_heads=12).
    """

    _layerwise_offload_blocks_attrs = ["blocks"]
    # Regional torch.compile target: the 30 identical DiT blocks are compiled once
    # and reused (see vllm_omni/diffusion/compile.py). Without this, compilation
    # falls back to a no-op regional pass.
    _repeated_blocks = ["CausalWanAttentionBlock"]

    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 1536,
        ffn_dim: int = 8960,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 12,
        num_layers: int = 30,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        num_frame_per_block: int = 1,
    ) -> None:
        super().__init__()
        if model_type != "t2v":
            raise ValueError(f"CausalForcing supports model_type='t2v' only, got {model_type!r}.")
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block

        # Disable the Conv3d GEMM rewrite for patch embedding.
        self.patch_embedding = Conv3dLayer(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding.enable_linear = False
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    "t2v_cross_attn",
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.head = CausalHead(dim, out_dim, patch_size, eps)

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        if (dim // num_heads) % 2 != 0:
            raise ValueError(f"dim // num_heads must be even, got {dim // num_heads}.")
        d = dim // num_heads
        # 3D RoPE frequency tables (f / h / w partition), matching Wan2.1.
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]

    def _create_freqs(self, grid_size: torch.Tensor, start_frame: int) -> torch.Tensor:
        """Build the 3D RoPE freqs tensor for a latent grid, offset by start_frame.

        The ``start_frame`` offset on the temporal axis is what makes attention
        causal across blocks: block N's tokens get absolute-frame positions.
        """
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]

        f, h, w = grid_size.tolist()
        freqs = torch.cat(
            [
                self.freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        return freqs

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
        """Reconstruct [B, C_out, F, H, W] video from patch embeddings."""
        B = x.shape[0]
        c = self.out_dim
        grid_size = grid_size.tolist()
        expected_seq_len = math.prod(grid_size)
        if x.shape[1] != expected_seq_len:
            raise ValueError(f"x sequence length must equal product(grid_size)={expected_seq_len}, got {x.shape[1]}.")
        x = x.view(B, *grid_size, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        kv_cache: list[dict],
        current_start: int,
        crossattn_cache: list[dict] | None = None,
    ) -> torch.Tensor:
        """Inference forward with the dict KV cache.

        Args:
            x: [B, C_in, F, H, W] latent block for the current chunk.
            timestep: [B, F] per-frame diffusion timestep.
            context: [B, text_len, text_dim] text embedding.
            kv_cache: per-layer self-attn cache dicts (mutated in place).
            current_start: absolute token offset (= start_frame * frame_seqlen).
            crossattn_cache: per-layer cross-attn cache dicts (mutated in place).

        Returns:
            flow_pred: [B, C_out, F, H, W] flow/velocity prediction.
        """
        if context.shape[1] != self.text_len:
            raise ValueError(f"context length must be {self.text_len}, got {context.shape[1]}.")

        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        f, h, w = grid_size.tolist()
        frame_seqlen = h * w
        seq_len = f * h * w
        current_start_frame = current_start // frame_seqlen
        freqs = self._create_freqs(grid_size, current_start_frame)

        x = x.flatten(start_dim=2).transpose(1, 2)  # [B, seq_len, dim]
        B = x.shape[0]
        F_t = timestep.shape[1]

        # Expand per-frame timestep to per-token and build modulation.
        timestep = timestep.unsqueeze(-1).expand(B, F_t, seq_len // F_t).reshape(B, -1)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))  # [B, seq_len, dim]
        e0 = self.time_projection(e).unflatten(dim=2, sizes=(6, self.dim))  # [B, seq_len, 6, dim]

        context = self.text_embedding(context)

        for block_index, block in enumerate(self.blocks):
            x = block(
                x=x,
                e=e0,
                freqs=freqs,
                frame_seqlen=frame_seqlen,
                context=context,
                kv_cache=kv_cache[block_index],
                crossattn_cache=crossattn_cache[block_index] if crossattn_cache else None,
                current_start=current_start,
            )

        x = self.head(x, e.unsqueeze(2))
        x = x.clone()
        flow_pred = self.unpatchify(x, grid_size)
        return flow_pred
