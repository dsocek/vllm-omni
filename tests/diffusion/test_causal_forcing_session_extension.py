# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol tests for CausalForcingSessionExtension.

These exercise the registry contract only -- lifecycle, capacity, role dispatch, and
the fail-loud paths -- against fake pipelines. No model, no device, so they run on
CPU in milliseconds. The rollout itself is covered by the pipeline's own tests; what
is worth pinning down here is that a mis-routed or duplicate call *fails* rather than
quietly starting a fresh shot, because that failure mode is invisible in the output.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from vllm_omni.diffusion.models.causal_forcing.session_extension import (
    CausalForcingSessionError,
    CausalForcingSessionExtension,
)


class FakeStream:
    def __init__(self, nfpb: int = 1) -> None:
        self.nfpb = nfpb
        self.scenes_played = 0
        self.latents_emitted = 0
        self.current_start_frame = 0
        self.pending_latents = 0
        self.closed = False
        self.pushed: list[tuple[str, dict]] = []
        self._queued_blocks = 0

    def push_scene(self, prompt, *, latents=21, transition="continue", seed=None):
        if not prompt.strip():
            raise ValueError("Scene prompt must be non-empty.")
        if transition not in ("continue", "cut"):
            raise ValueError(f"transition must be 'continue' or 'cut', got {transition!r}.")
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be an int or None, got {type(seed).__name__}.")
        self.pushed.append((prompt, {"latents": latents, "transition": transition, "seed": seed}))
        self._queued_blocks += latents // self.nfpb
        self.pending_latents += latents

    def blocks(self):
        # Mirrors the real generator's contract: drains the queue, then returns (does
        # not raise StopIteration early), so the caller can push more and re-enter.
        while self._queued_blocks:
            self._queued_blocks -= 1
            self.latents_emitted += self.nfpb
            self.current_start_frame += self.nfpb
            self.pending_latents -= self.nfpb
            yield f"block{self.latents_emitted}"
        self.scenes_played += 1

    def close(self):
        self.closed = True


class FakeTransformer:
    local_attn_size = 21


class FakeDiTPipeline:
    _stage_role = "dit"

    def __init__(self) -> None:
        self.transformer = FakeTransformer()
        self.streams: list[FakeStream] = []

    def open_stream(self, *, height=480, width=832, seed=None):
        stream = FakeStream()
        stream.seed = seed
        self.streams.append(stream)
        return stream


class FakeVAEPipeline:
    _stage_role = "vae"

    def __init__(self) -> None:
        self.begins = 0
        self.ends = 0
        self.decoded: list[str] = []

    def open_stream(self, **kwargs):
        raise RuntimeError("open_stream requires the DiT role; the VAE stage has no rollout to stream.")

    def _decode_stream_begin(self):
        self.begins += 1

    def _decode_stream_step(self, latents):
        self.decoded.append(latents)
        return f"pixels({latents})"

    def _decode_stream_end(self):
        self.ends += 1


class FakeModelRunner:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline


class FakeWorker(CausalForcingSessionExtension):
    """Stands in for the worker the mixin is grafted onto by WorkerWrapperBase."""

    def __init__(self, pipeline) -> None:
        self.model_runner = FakeModelRunner(pipeline)


@pytest.fixture
def dit():
    return FakeWorker(FakeDiTPipeline())


@pytest.fixture
def vae():
    return FakeWorker(FakeVAEPipeline())


# -- DiT role lifecycle -----------------------------------------------------


def test_open_push_drain_close(dit):
    info = dit.session_open("s1")
    assert info["role"] == "dit"
    assert info["local_attn_size"] == 21

    dit.session_push("s1", "a coral reef", latents=3, transition="cut")
    assert dit.session_info("s1")["pending_latents"] == 3

    chunks = dit.session_drain("s1")
    assert len(chunks) == 3
    assert dit.session_info("s1")["latents_emitted"] == 3

    closed = dit.session_close("s1")
    assert closed["closed"] is True
    assert dit.model_runner.pipeline.streams[0].closed is True


def test_drain_is_reenterable_across_pushes(dit):
    """The whole point of a session: scene n+1 continues scene n's rollout."""
    dit.session_open("s1")
    dit.session_push("s1", "scene one", latents=2)
    assert len(dit.session_drain("s1")) == 2
    dit.session_push("s1", "scene two", latents=2)
    assert len(dit.session_drain("s1")) == 2
    # Same stream throughout -- a new one would mean the KV window was discarded.
    assert len(dit.model_runner.pipeline.streams) == 1
    assert dit.session_info("s1")["latents_emitted"] == 4


def test_drain_respects_max_blocks(dit):
    dit.session_open("s1")
    dit.session_push("s1", "long scene", latents=5)
    assert len(dit.session_drain("s1", max_blocks=2)) == 2
    # The rest stays queued rather than being dropped.
    assert dit.session_info("s1")["pending_latents"] == 3
    assert len(dit.session_drain("s1")) == 3


def test_transition_and_latents_reach_the_stream(dit):
    dit.session_open("s1")
    dit.session_push("s1", "a cut", latents=4, transition="cut")
    dit.session_push("s1", "a continue")
    stream = dit.model_runner.pipeline.streams[0]
    assert stream.pushed[0][1] == {"latents": 4, "transition": "cut", "seed": None}
    # latents omitted -> the stream's own default applies, not one invented here.
    assert stream.pushed[1][1]["transition"] == "continue"
    assert stream.pushed[1][1]["latents"] == 21


# -- seeds ------------------------------------------------------------------


def test_stream_seed_reaches_open_stream(dit):
    dit.session_open("s1", seed=1234)
    assert dit.model_runner.pipeline.streams[0].seed == 1234


def test_scene_seed_reaches_push_scene(dit):
    dit.session_open("s1")
    dit.session_push("s1", "a re-rolled shot", seed=99)
    assert dit.model_runner.pipeline.streams[0].pushed[0][1]["seed"] == 99


def test_scene_seed_zero_is_forwarded_not_dropped(dit):
    """0 is a real seed. The forwarding is `is not None`, not truthiness -- a falsy
    check here would silently fall back to the stream's derived seed."""
    dit.session_open("s1")
    dit.session_push("s1", "shot", seed=0)
    assert dit.model_runner.pipeline.streams[0].pushed[0][1]["seed"] == 0


def test_omitted_scene_seed_leaves_derivation_to_the_stream(dit):
    """The extension must not invent a seed: an unset one has to reach push_scene as
    None so the stream derives it from the stream seed and the scene's ordinal."""
    dit.session_open("s1", seed=7)
    dit.session_push("s1", "shot")
    assert dit.model_runner.pipeline.streams[0].pushed[0][1]["seed"] is None


def test_bad_scene_seed_surfaces_as_a_protocol_error(dit):
    dit.session_open("s1")
    with pytest.raises(CausalForcingSessionError, match="seed must be an int"):
        dit.session_push("s1", "shot", seed="99")


# -- fail-loud paths --------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda w: w.session_push("ghost", "prompt"),
        lambda w: w.session_drain("ghost"),
        lambda w: w.session_close("ghost"),
        lambda w: w.session_info("ghost"),
    ],
)
def test_unknown_session_is_an_error_not_a_fresh_stream(dit, call):
    dit.session_open("s1")
    with pytest.raises(CausalForcingSessionError, match="ghost"):
        call(dit)
    # Nothing was created as a side effect of the failed lookup.
    assert len(dit.model_runner.pipeline.streams) == 1


def test_duplicate_open_is_rejected(dit):
    dit.session_open("s1")
    with pytest.raises(CausalForcingSessionError, match="already open"):
        dit.session_open("s1")
    assert len(dit.model_runner.pipeline.streams) == 1


def test_capacity_limit_is_enforced(dit):
    dit.session_open("s1")
    with pytest.raises(CausalForcingSessionError, match="session limit"):
        dit.session_open("s2")
    dit.session_close("s1")
    dit.session_open("s2")  # slot freed
    assert dit.session_info()["open_sessions"] == 1


def test_capacity_limit_is_raisable(dit):
    dit.session_open("s1", max_sessions=2)
    dit.session_open("s2", max_sessions=2)
    assert dit.session_info()["open_sessions"] == 2


def test_empty_session_id_rejected(dit):
    with pytest.raises(CausalForcingSessionError, match="non-empty"):
        dit.session_open("")


def test_push_validation_surfaces_as_protocol_error(dit):
    dit.session_open("s1")
    with pytest.raises(CausalForcingSessionError, match="rejected"):
        dit.session_push("s1", "   ")
    with pytest.raises(CausalForcingSessionError, match="rejected"):
        dit.session_push("s1", "ok", transition="dissolve")


def test_close_missing_ok(dit):
    assert dit.session_close("ghost", missing_ok=True)["closed"] is False


def test_no_pipeline_is_an_error():
    worker = FakeWorker(None)
    with pytest.raises(CausalForcingSessionError, match="No pipeline"):
        worker.session_open("s1")


def test_wrong_pipeline_type_is_an_error():
    class NotCF:
        pass

    worker = FakeWorker(NotCF())
    with pytest.raises(CausalForcingSessionError, match="open_stream"):
        worker.session_open("s1")


# -- VAE role ---------------------------------------------------------------


def test_vae_session_is_a_decode_cursor(vae):
    info = vae.session_open("s1")
    assert info["role"] == "vae"
    # Cache reset exactly once at open, so chunk 1 is first_chunk and later chunks
    # inherit its temporal context.
    assert vae.model_runner.pipeline.begins == 1
    assert "nfpb" not in info

    assert vae.session_decode_step("s1", "lat1") == "pixels(lat1)"
    assert vae.session_decode_step("s1", "lat2") == "pixels(lat2)"
    assert vae.model_runner.pipeline.begins == 1  # not re-begun mid-stream
    assert vae.session_info("s1")["chunks_decoded"] == 2

    vae.session_close("s1")
    assert vae.model_runner.pipeline.ends == 1


def test_rollout_calls_on_vae_worker_fail(vae):
    vae.session_open("s1")
    with pytest.raises(CausalForcingSessionError, match="only decodes"):
        vae.session_push("s1", "a prompt")
    with pytest.raises(CausalForcingSessionError, match="only decodes"):
        vae.session_drain("s1")


def test_decode_step_on_dit_worker_fails(dit):
    dit.session_open("s1")
    with pytest.raises(CausalForcingSessionError, match="VAE-role"):
        dit.session_decode_step("s1", "lat1")


# -- idle expiry ------------------------------------------------------------


def _age(worker, session_id, seconds):
    """Backdate a session's last-use stamp instead of sleeping, so the expiry tests
    assert on the threshold rather than on how fast the host happens to be."""
    worker._sessions[session_id].last_used_at -= seconds


def test_idle_session_expires_to_free_capacity(dit):
    dit.session_open("s1")
    # An abandoned session pins a KV window with no connection to signal its loss,
    # so time-since-use is the only liveness signal available at this layer.
    _age(dit, "s1", 120)
    dit.session_open("s2", idle_timeout_s=60)
    assert dit.session_info()["open_sessions"] == 1
    assert dit.model_runner.pipeline.streams[0].closed is True


def test_session_within_timeout_does_not_expire(dit):
    dit.session_open("s1")
    _age(dit, "s1", 30)
    with pytest.raises(CausalForcingSessionError, match="session limit"):
        dit.session_open("s2", idle_timeout_s=60)
    assert dit.model_runner.pipeline.streams[0].closed is False


def test_expiry_is_disabled_by_nonpositive_timeout(dit):
    dit.session_open("s1")
    _age(dit, "s1", 10_000)
    with pytest.raises(CausalForcingSessionError, match="session limit"):
        dit.session_open("s2", idle_timeout_s=0)
    assert dit.model_runner.pipeline.streams[0].closed is False


def test_failed_teardown_still_frees_the_slot(dit):
    dit.session_open("s1")

    def boom():
        raise RuntimeError("cache release failed")

    dit.model_runner.pipeline.streams[0].close = boom
    _age(dit, "s1", 120)
    dit.session_open("s2", idle_timeout_s=60)
    assert dit.session_info()["open_sessions"] == 1


# -- deploy wiring ----------------------------------------------------------

_DEPLOY_YAML = pathlib.Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy" / "causal_forcing_disagg.yaml"
_EXTENSION_QUALNAME = "vllm_omni.diffusion.models.causal_forcing.session_extension.CausalForcingSessionExtension"


@pytest.mark.skipif(not _DEPLOY_YAML.exists(), reason="deploy yaml not present")
def test_deploy_yaml_declares_the_extension_on_both_stages():
    """The qualname is a string in YAML, so nothing else would catch a typo or a
    rename until a worker tried to resolve it at startup."""
    from vllm.utils.import_utils import resolve_obj_by_qualname

    stages = yaml.safe_load(_DEPLOY_YAML.read_text())["stages"]
    for stage in stages:
        assert stage.get("worker_extension_cls") == _EXTENSION_QUALNAME, (
            f"stage {stage.get('stage_id')} is missing the session extension"
        )
    assert resolve_obj_by_qualname(_EXTENSION_QUALNAME) is CausalForcingSessionExtension


@pytest.mark.skipif(not _DEPLOY_YAML.exists(), reason="deploy yaml not present")
def test_deploy_yaml_uses_a_finite_attention_window():
    """open_stream() rejects local_attn_size=-1, so the streaming deploy must declare
    a finite window or every session_open would fail at runtime."""
    stages = yaml.safe_load(_DEPLOY_YAML.read_text())["stages"]
    dit = next(s for s in stages if s.get("model_class_name") == "CausalForcingDiTPipeline")
    assert dit["model_config"]["local_attn_size"] > 0


def test_extension_does_not_collide_with_the_worker_it_mixes_onto():
    """WorkerWrapperBase builds type(name, (extension, worker), {}), so the extension
    wins every name clash -- silently shadowing a worker method."""
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    public = [a for a in dir(CausalForcingSessionExtension) if not a.startswith("__")]
    assert public, "sanity: the extension should expose something"
    clashes = [a for a in public if hasattr(DiffusionWorker, a)]
    assert clashes == []


def test_extension_defines_no_init():
    """The mixin is grafted on ahead of the worker in the MRO; an __init__ here would
    displace the worker's own constructor. Hence the lazily-attached registry."""
    assert "__init__" not in vars(CausalForcingSessionExtension)
