# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spatially-sharded streaming VAE decode for Causal-Forcing.

The streaming decode and patch parallelism used to be mutually exclusive: streaming
called ``self.vae.decoder`` directly, which bypassed the ``is_distributed_enabled()``
hooks that live on ``decode()``/``tiled_decode()``, so ``vae_patch_parallel_size>1``
silently fell back to a whole-clip decode. Whole-clip is exactly what a pipelined
stream cannot use -- it cannot start until the last latent frame exists, which is the
wait the pipelining removes.

What makes the two composable is *which axis* is split. Tiles are a slice of the clip;
a spatial shard is a stripe of one frame. So the tests here pin two things:

* ``tile`` mode must NOT be treated as shardable-while-streaming, and
* a sharded stream must install the decoder patch exactly once and then keep every
  rank in lockstep, because a rank that skipped a chunk would leave the others
  blocked in a halo exchange -- a hang, not an error.

No model and no device: the VAE is a stub that records the calls it receives.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.causal_forcing.pipeline_causal_forcing import (
    CausalForcingPipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def fake_process_group(monkeypatch):
    """A 2-rank group with this process as rank 0, unless a test says otherwise.

    ``_sharded_stream_bystander`` asks the real ``torch.distributed`` for the group's
    size, and the stub's group is a bare sentinel, so every sharded test needs this --
    not just the ones about rank roles.
    """
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    return monkeypatch


class StubDecoder:
    """Records the (frame, first_chunk) pairs it was asked to decode."""

    def __init__(self, out_hw: tuple[int, int] = (8, 8)) -> None:
        self.calls: list[bool] = []
        self.out_hw = out_hw

    def __call__(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        self.calls.append(first_chunk)
        b, _, t, _, _ = x.shape
        return torch.zeros((b, 3, t, *self.out_hw))


class StubVae:
    """The surface ``_decode_stream_*`` touches, and nothing else."""

    def __init__(self, parallel_mode: str = "tile", parallel_size: int = 2, world_size: int = 2) -> None:
        self.decoder = StubDecoder()
        self.config = SimpleNamespace(patch_size=None)
        self.dtype = torch.float32
        self._feat_map = [None]
        self._conv_idx = [0]
        self.cache_cleared = 0
        self.distributed_executor = SimpleNamespace(
            group=object(),
            parallel_size=parallel_size,
            parallel_mode=parallel_mode,
        )
        self._world_size = world_size

    def clear_cache(self) -> None:
        self.cache_cleared += 1

    def post_quant_conv(self, x):
        return x

    def parameters(self):
        yield torch.zeros(1)

    def _execution_context(self):
        import contextlib

        return contextlib.nullcontext()

    def spatial_shard_decode_split_dim_if_ready(self):
        mode = self.distributed_executor.parallel_mode
        if not mode.startswith("spatial_shard_"):
            return None
        if self.distributed_executor.parallel_size != self._world_size:
            return None
        return mode.removeprefix("spatial_shard_")


def _pipeline(vae: StubVae) -> CausalForcingPipeline:
    """A pipeline shell holding only what the decode path reads."""
    pipe = CausalForcingPipeline.__new__(CausalForcingPipeline)
    pipe.vae = vae
    pipe.vae_latents_mean = torch.zeros((1, 4, 1, 1, 1))
    pipe.vae_latents_inv_std = torch.ones((1, 4, 1, 1, 1))
    return pipe


def _latents(frames: int = 1) -> torch.Tensor:
    return torch.zeros((1, 4, frames, 6, 6))


# -- the probe: which modes are shardable while streaming -------------------


def test_tile_mode_is_not_shardable_while_streaming():
    """A tile is a slice of the *clip*, so there is nothing to shard inside one frame.
    Reading tile mode as shardable would install a halo-exchanging decoder for a split
    that never happens."""
    pipe = _pipeline(StubVae(parallel_mode="tile"))

    assert pipe._stream_shard_split_dim() is None


@pytest.mark.parametrize("axis", ["height", "width"])
def test_spatial_shard_modes_are_shardable_while_streaming(axis):
    pipe = _pipeline(StubVae(parallel_mode=f"spatial_shard_{axis}"))

    assert pipe._stream_shard_split_dim() == axis


def test_a_partial_shard_group_is_not_shardable():
    """vae_patch_parallel_size must match the group; the VAE's own gate says so, and the
    stream must not reach a different conclusion than the whole-clip path does."""
    pipe = _pipeline(StubVae(parallel_mode="spatial_shard_height", parallel_size=2, world_size=4))

    assert pipe._stream_shard_split_dim() is None


def test_a_vae_without_the_gate_is_not_shardable():
    """A plain (non-distributed) VAE has no such method. Streaming has to keep working."""

    class PlainVae(StubVae):
        # None, not deleted: ``getattr(..., None)`` is what the probe uses, so absent
        # and present-but-None have to answer the same way.
        spatial_shard_decode_split_dim_if_ready = None

    pipe = _pipeline(PlainVae())

    assert pipe._stream_shard_split_dim() is None


# -- installing the patch --------------------------------------------------


def test_a_sharded_stream_installs_the_decoder_patch_once(monkeypatch):
    """Once per stream, at begin -- not per chunk. Installing mutates the decoder, and
    every rank has to agree for every chunk or the halo exchange deadlocks."""
    installs: list[str] = []
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard"
        ".install_wan_spatial_shard_decode",
        lambda vae, group, split_dim="height": installs.append(split_dim),
    )
    pipe = _pipeline(StubVae(parallel_mode="spatial_shard_height"))

    pipe._decode_stream_begin()
    pipe._decode_stream_step(_latents())
    pipe._decode_stream_step(_latents())

    assert installs == ["height"]


def test_an_unsharded_stream_installs_nothing(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard"
        ".install_wan_spatial_shard_decode",
        lambda *a, **k: pytest.fail("a single-rank stream must not patch the decoder"),
    )
    pipe = _pipeline(StubVae(parallel_mode="tile"))

    pipe._decode_stream_begin()
    pipe._decode_stream_step(_latents())


# -- the temporal chain survives sharding ----------------------------------


def test_first_chunk_is_true_only_on_the_very_first_frame(monkeypatch):
    """The property that makes chunk boundaries seam-free, and it must not change just
    because the frame is now split across ranks: sharding divides space, and the cache
    it has to leave intact runs along time."""
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard"
        ".install_wan_spatial_shard_decode",
        lambda *a, **k: None,
    )
    vae = StubVae(parallel_mode="spatial_shard_height")
    pipe = _pipeline(vae)

    pipe._decode_stream_begin()
    pipe._decode_stream_step(_latents(frames=2))
    pipe._decode_stream_step(_latents(frames=2))

    assert vae.decoder.calls == [True, False, False, False]


def test_every_latent_frame_reaches_the_decoder(monkeypatch):
    """A dropped frame would shorten the clip rather than fail, so count them."""
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard"
        ".install_wan_spatial_shard_decode",
        lambda *a, **k: None,
    )
    vae = StubVae(parallel_mode="spatial_shard_height")
    pipe = _pipeline(vae)

    pipe._decode_stream_begin()
    out = pipe._decode_stream_step(_latents(frames=3))

    assert len(vae.decoder.calls) == 3
    assert out.shape[2] == 3


# -- rank 0 assembles; the others only keep the collectives moving ---------


def _sharded_pipe(monkeypatch, *, rank: int, world_size: int = 2):
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard"
        ".install_wan_spatial_shard_decode",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "torch.distributed.get_world_size", lambda group=None: world_size
    )
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: rank)
    pipe = _pipeline(StubVae(parallel_mode="spatial_shard_height", world_size=world_size, parallel_size=world_size))
    pipe._decode_stream_begin()
    return pipe


def test_rank_zero_returns_the_assembled_frames(monkeypatch):
    pipe = _sharded_pipe(monkeypatch, rank=0)

    out = pipe._decode_stream_step(_latents(frames=2))

    assert out.ndim == 5 and out.shape[2] == 2


def test_a_bystander_rank_returns_an_empty_placeholder(monkeypatch):
    """Not its shard. A shard is a stripe of an image, and handing one back as though it
    were a frame is the kind of mistake that survives all the way into a video file."""
    pipe = _sharded_pipe(monkeypatch, rank=1)

    out = pipe._decode_stream_step(_latents(frames=2))

    assert out.numel() == 0


def test_a_bystander_still_runs_every_decoder_call(monkeypatch):
    """It must stay in lockstep: the halo exchanges and all-gathers are collective, so a
    rank that skipped a chunk would hang the ones that did not."""
    pipe = _sharded_pipe(monkeypatch, rank=1)

    pipe._decode_stream_step(_latents(frames=3))

    assert len(pipe.vae.decoder.calls) == 3


def test_concat_tolerates_the_bystander_placeholders():
    """The aggregated rollout concatenates its own chunks, so on a bystander rank it
    would otherwise concat along a dimension the placeholders do not have."""
    real = torch.zeros((1, 3, 2, 8, 8))
    empty = torch.zeros(0)

    assert CausalForcingPipeline._concat_stream_chunks([real, real]).shape[2] == 4
    assert CausalForcingPipeline._concat_stream_chunks([empty, empty]).numel() == 0


def test_single_rank_decode_is_never_a_bystander(monkeypatch):
    """world=1 means rank 0 is the only rank; it must assemble, not defer."""
    pipe = _sharded_pipe(monkeypatch, rank=0, world_size=1)

    assert pipe._sharded_stream_bystander() is False
