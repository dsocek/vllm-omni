# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side session registry for open-ended Causal-Forcing streams.

``CausalForcingStream`` (see :mod:`.pipeline_causal_forcing`) holds a live KV
window, a VAE temporal cache, and a frame cursor — all device-resident. It
therefore cannot cross a process boundary, and it has to outlive the request that
created it. Neither the Dynamo stage worker nor the vLLM-Omni entrypoint can hold
it: the pipeline lives one process further down, inside ``WorkerProc``.

This module puts the registry where the pipeline actually is. It is a
``worker_extension_cls`` mixin — the same in-tree mechanism
``CustomPipelineWorkerExtension`` uses — so its methods become callable via
``DiffusionEngine.collective_rpc`` / ``async_collective_rpc``.

Two properties make that the right transport rather than merely a convenient one:

* ``collective_rpc`` calls are drained by the engine's busy loop *between*
  scheduler steps, so a session mutation can never interleave with a rollout
  block that is mid-flight. The registry needs no lock of its own.
* The extension is mixed onto the worker, so ``self.model_runner.pipeline`` is
  the real pipeline object — no serialization, no copy of the KV window.

Protocol, all keyed by an opaque ``session_id`` (Dynamo supplies
``x-dynamo-session-id``):

    session_open(session_id, ...)      -> dict   fails if that id is already open
    session_push(session_id, prompt)   -> dict   queue a scene (text encode only)
    session_drain(session_id, ...)     -> list   roll out queued blocks
    session_close(session_id)          -> dict   release
    session_info(session_id=None)      -> dict   introspection / capacity

The same class serves both roles of the disaggregated deploy, because which role a
worker plays is a property of the pipeline it loaded, not of the extension:

* **DiT role** (stage 0) owns the rollout. ``open_stream`` builds the
  ``CausalForcingStream``, and because that pipeline has no VAE, ``blocks()`` yields
  *model-space latents* — the same tensors ``forward`` would hand the ``dit2vae``
  bridge. It does not yield pixels.
* **VAE role** (stage 1) has no rollout at all; ``open_stream`` refuses it outright.
  What it owns is the decoder's temporal ``feat_cache``, which is what keeps chunk
  boundaries seam-free. So its session is a *decode cursor*: begin once, step per
  chunk, end on close, via ``session_decode_step``.

One consequence worth stating plainly: the VAE's ``feat_cache`` lives on the VAE
module itself, not per-session, so a decoding session is necessarily **exclusive**
on its worker. On that role the ceiling of 1 is a *correctness* bound, not a tuning
knob set low, and ``session_open`` clamps to it regardless of what the caller asks
for. Scaling the VAE out means more replicas -- each its own process, hence its own
``feat_cache`` -- not a bigger ceiling.

The DiT role is the opposite case: every session there owns its own
``CausalForcingStream`` and KV window, so the ceiling is a *memory* bound and a card
with room can hold several at once. That is what lets one DiT feed several VAE
replicas concurrently, so it is settable -- see ``CF_MAX_SESSIONS``.

A mis-routed call is a hard error, never a silently-started fresh stream. For an
LLM a routing miss costs prefix-cache reuse; here the KV window is the only copy
of the shot in progress and it is pinned to one card, so guessing would emit a
visibly broken video instead of failing.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.models.causal_forcing.pipeline_causal_forcing import CausalForcingStream

logger = init_logger(__name__)

# A session pins a KV window (~16 GB at 480x832) and its DiT+VAE pair for as long
# as it is open, so an abandoned one is expensive. Callers that want a different
# ceiling pass max_sessions to session_open.
#
# Settable because on the DiT role this is a memory bound, and one DiT can only feed
# N VAE replicas concurrently if it will hold N sessions at once -- with the default
# of 1, a second concurrent stream is refused at session_open and scene parallelism
# is unreachable no matter how many VAE replicas are deployed.
#
# It stays 1 by default because the safe value depends on the card: at ~16 GB per KV
# window, an 80 GB card holds about 4 alongside weights, and overcommitting shows up
# as an OOM mid-rollout rather than a clean refusal. Set it deliberately, per host.
#
# The VAE role IGNORES this -- see session_open, where the cap resolves per role, and
# the module docstring for why 1 is a correctness bound there.
_MAX_SESSIONS_ENV = "CF_MAX_SESSIONS"


def _default_max_sessions() -> int:
    """Read the DiT-role session ceiling from the environment, defaulting to 1."""
    raw = os.environ.get(_MAX_SESSIONS_ENV, "").strip()
    if not raw:
        return 1
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "[CF_SESSION] ignoring %s=%r: not an integer; using 1",
            _MAX_SESSIONS_ENV,
            raw,
        )
        return 1
    if value < 1:
        logger.warning(
            "[CF_SESSION] ignoring %s=%d: must be >= 1; using 1",
            _MAX_SESSIONS_ENV,
            value,
        )
        return 1
    return value


DEFAULT_MAX_SESSIONS = _default_max_sessions()

# Sessions idle longer than this expire on the next registry touch. A stream
# has no connection to watch, so time since last use is the only liveness signal
# available down here.
#
# This is also the *implicit* close: a client that stops sending scenes has its shot
# released without ever saying so. ``cf_close`` only makes that immediate. The value is
# a trade -- long enough that a user thinking about the next prompt does not lose their
# KV window, short enough that an abandoned session does not hold a card all day.
DEFAULT_IDLE_TIMEOUT_S = 600.0


class CausalForcingSessionError(RuntimeError):
    """Raised for protocol misuse: unknown, duplicate, or closed session."""


class _Session:
    """One live session plus the bookkeeping the registry needs to age it out.

    ``stream`` is None on the VAE role: that stage has no rollout to hold, only the
    decoder's temporal cache, which lives on the VAE module.
    """

    def __init__(self, session_id: str, role: str, stream: CausalForcingStream | None) -> None:
        self.session_id = session_id
        self.role = role
        self.stream = stream
        self.created_at = time.monotonic()
        self.last_used_at = self.created_at
        self.scenes_pushed = 0
        self.blocks_emitted = 0
        self.chunks_decoded = 0

    def touch(self) -> None:
        self.last_used_at = time.monotonic()

    def idle_for(self) -> float:
        return time.monotonic() - self.last_used_at

    def stats(self) -> dict[str, Any]:
        stats = {
            "session_id": self.session_id,
            "role": self.role,
            "age_s": round(time.monotonic() - self.created_at, 1),
            "idle_s": round(self.idle_for(), 1),
        }
        if self.stream is None:
            stats["chunks_decoded"] = self.chunks_decoded
            return stats
        stream = self.stream
        stats.update(
            scenes_pushed=self.scenes_pushed,
            scenes_played=stream.scenes_played,
            blocks_emitted=self.blocks_emitted,
            latents_emitted=stream.latents_emitted,
            pending_latents=stream.pending_latents,
            current_start_frame=stream.current_start_frame,
        )
        return stats

    def require_stream(self, method: str) -> CausalForcingStream:
        if self.stream is None:
            raise CausalForcingSessionError(
                f"{method}() needs a rollout, but session {self.session_id!r} is a "
                f"{self.role} session, which only decodes."
            )
        return self.stream


class CausalForcingSessionExtension:
    """Worker mixin exposing a ``session_id`` -> live-stream registry over RPC.

    Mixed onto the diffusion worker via ``worker_extension_cls``, so ``self`` is
    the worker and ``self.model_runner.pipeline`` is the loaded pipeline. State is
    created lazily because the mixin has no ``__init__`` of its own — the worker
    class owns construction.
    """

    # -- registry plumbing --------------------------------------------------

    @property
    def _sessions(self) -> dict[str, _Session]:
        # Lazily attached: a mixin must not define __init__, or it would have to
        # cooperate with the worker's own constructor chain.
        registry = getattr(self, "_cf_sessions", None)
        if registry is None:
            registry = {}
            self._cf_sessions = registry
        return registry

    def _pipeline(self) -> Any:
        pipeline = getattr(getattr(self, "model_runner", None), "pipeline", None)
        if pipeline is None:
            raise CausalForcingSessionError("No pipeline loaded on this worker; cannot serve a session.")
        if not hasattr(pipeline, "open_stream"):
            raise CausalForcingSessionError(
                f"Pipeline {type(pipeline).__name__} has no open_stream(); "
                "session streaming needs a CausalForcing pipeline."
            )
        return pipeline

    def _stage_role(self) -> str:
        """``"dit"``, ``"vae"``, or ``"full"`` — which half of the deploy this worker is."""
        return getattr(self._pipeline(), "_stage_role", "full")

    def _require_role(self, method: str, *roles: str) -> Any:
        """Reject a call aimed at the wrong stage, naming both roles.

        Worth being explicit rather than letting it fail deeper: a rollout call that
        lands on the VAE worker means the router mapped the session to the wrong half
        of the pipeline, and that diagnosis is much cheaper to read here than as an
        AttributeError inside a decode loop.
        """
        pipeline = self._pipeline()
        role = getattr(pipeline, "_stage_role", "full")
        if role not in roles:
            raise CausalForcingSessionError(
                f"{method}() requires the {' or '.join(roles)} role, but this worker loaded "
                f"{type(pipeline).__name__} (role={role!r}). The session was routed to the "
                "wrong stage of the disaggregated pipeline."
            )
        return pipeline

    def _get(self, session_id: str) -> _Session:
        """Look up a session, or fail loudly.

        This is the fail-loud boundary. Starting a fresh stream for an unknown id
        would silently reset the shot — the client would receive a video that cuts
        back to noise with no error anywhere.
        """
        session = self._sessions.get(session_id)
        if session is None:
            known = sorted(self._sessions)
            raise CausalForcingSessionError(
                f"Unknown session {session_id!r} on this worker (open: {known or 'none'}). "
                "A session is pinned to the worker that opened it; this request was routed "
                "elsewhere or the session was already closed or expired."
            )
        return session

    def _expire_idle(self, idle_timeout_s: float) -> list[str]:
        """Close sessions nobody has touched lately. Returns the ids that expired.

        This is how a session ends when the client never says so: it stops sending
        scenes, the idle time crosses the timeout, and the next call that touches the
        registry releases the KV window. ``session_close`` is the same ending, just
        immediate. Called from ``session_open`` because that is the moment capacity is
        actually needed — there is no background task down here, and a worker with no
        traffic has nothing to reclaim capacity *for*.
        """
        if idle_timeout_s <= 0:
            return []
        stale = [sid for sid, s in self._sessions.items() if s.idle_for() > idle_timeout_s]
        for sid in stale:
            session = self._sessions.pop(sid)
            logger.warning(
                "[CF_SESSION] expiring idle %s session %s after %.0fs",
                session.role,
                sid,
                session.idle_for(),
            )
            try:
                self._teardown(session)
            except Exception:
                # Never let a failed teardown strand the slot: the session is already
                # out of the registry, so the capacity it held must be released.
                logger.exception("[CF_SESSION] error closing idle session %s", sid)
        return stale

    def _teardown(self, session: _Session) -> None:
        """Release whichever caches this role owns."""
        if session.stream is not None:
            session.stream.close()
        else:
            self._pipeline()._decode_stream_end()

    # -- protocol -----------------------------------------------------------

    def session_open(
        self,
        session_id: str,
        *,
        height: int = 480,
        width: int = 832,
        seed: int | None = None,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Open a stream for ``session_id``. Fails if that id is already open."""
        if not session_id:
            raise CausalForcingSessionError("session_id must be a non-empty string.")

        expired = self._expire_idle(idle_timeout_s)

        if session_id in self._sessions:
            # Not idempotent on purpose: a duplicate open means the caller thinks
            # it is starting a new shot while a rollout it does not know about is
            # still live. Reopening would discard that history silently.
            raise CausalForcingSessionError(
                f"Session {session_id!r} is already open on this worker. "
                "Close it before reopening, or push scenes onto the existing session."
            )

        pipeline = self._pipeline()
        role = getattr(pipeline, "_stage_role", "full")

        # The ceiling means two different things per role, so it resolves per role.
        #
        # On the VAE role it is a CORRECTNESS bound and the caller does not get a
        # vote: feat_cache lives on the VAE module rather than on the session, so a
        # second concurrent decode cursor would interleave into the first one's
        # temporal context and emit seams with nothing in the output to say so.
        # Capacity there is added by running more replicas, each its own process.
        #
        # On the DiT role each session owns a separate CausalForcingStream and KV
        # window, so the bound is only memory and several may be open at once. That
        # is what lets one DiT feed several VAE replicas concurrently.
        if role == "vae":
            effective_max = 1
            limit_reason = (
                "the VAE's feat_cache is per-module, not per-session, so one decode "
                "cursor per worker process is the only safe value; add VAE replicas "
                "to decode more streams at once"
            )
        else:
            effective_max = max_sessions
            limit_reason = (
                f"each session pins its own KV window, so this is a memory bound; "
                f"raise {_MAX_SESSIONS_ENV} if the card has room"
            )

        if len(self._sessions) >= effective_max:
            raise CausalForcingSessionError(
                f"Worker is at its {role} session limit "
                f"({len(self._sessions)}/{effective_max}). "
                f"Open sessions: {sorted(self._sessions)}. {limit_reason}."
            )

        info: dict[str, Any] = {
            "session_id": session_id,
            "role": role,
            "height": height,
            "width": width,
            "seed": seed,
            "expired_idle": expired,
        }

        if role == "vae":
            # No rollout here. Opening a session means resetting the decoder's temporal
            # cache exactly once, so the first chunk is treated as first_chunk and every
            # later one carries the previous chunk's context forward.
            pipeline._decode_stream_begin()
            stream = None
        else:
            stream = pipeline.open_stream(height=height, width=width, seed=seed)
            info["nfpb"] = stream.nfpb
            info["local_attn_size"] = pipeline.transformer.local_attn_size

        self._sessions[session_id] = _Session(session_id, role, stream)
        logger.info(
            "[CF_SESSION] opened %s role=%s (%dx%d seed=%s) — %d session(s) on this worker",
            session_id,
            role,
            height,
            width,
            seed,
            len(self._sessions),
        )
        return info

    def session_push(
        self,
        session_id: str,
        prompt: str,
        *,
        latents: int | None = None,
        transition: str = "continue",
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Queue one scene. Cheap: text encode only, no rollout.

        ``seed`` pins this scene's noise so the shot can be reproduced or re-rolled on
        its own. Exact for a ``cut``; for a ``continue`` it reproduces the scene given
        the same preceding scenes, since a continue also attends to their KV history.
        """
        session = self._get(session_id)
        stream = session.require_stream("session_push")
        session.touch()

        kwargs: dict[str, Any] = {"transition": transition}
        if latents is not None:
            kwargs["latents"] = latents
        if seed is not None:
            kwargs["seed"] = seed
        try:
            stream.push_scene(prompt, **kwargs)
        except (ValueError, RuntimeError) as e:
            # push_scene validates prompt/transition/length. Surface it as a
            # protocol error so the caller sees a 4xx-shaped failure, not a crash.
            raise CausalForcingSessionError(f"session_push({session_id!r}) rejected: {e}") from e

        session.scenes_pushed += 1
        logger.info(
            "[CF_SESSION] %s queued scene %d (%s, %d pending latents): %.60s",
            session_id,
            session.scenes_pushed,
            transition,
            stream.pending_latents,
            prompt,
        )
        return {
            "session_id": session_id,
            "scenes_pushed": session.scenes_pushed,
            "pending_latents": stream.pending_latents,
        }

    def session_drain(self, session_id: str, *, max_blocks: int | None = None) -> list[torch.Tensor]:
        """Roll out queued blocks, one tensor per block.

        Returns when the queue empties, or after ``max_blocks`` — the latter lets a
        caller interleave pushes with generation instead of committing to drain
        everything queued so far.

        **What comes back depends on the role**, because it is whatever
        ``CausalForcingStream.blocks()`` yields:

        * DiT stage (the disaggregated deploy) has no VAE, so these are model-space
          *latents* ``[B, C, nfpb, H/8, W/8]`` — feed them to the VAE stage's
          ``session_decode_step``, exactly as the ``dit2vae`` bridge does for one-shot.
        * Aggregated (``role="full"``) pipelines decode inline, so these are pixels
          ``[B, C, t, H, W]``.

        The tensors are returned as-is, but note this is **not** a zero-copy device
        handoff: ``collective_rpc`` results are pickled across the worker process
        boundary, so each block takes a full host round-trip (measured ~585 KB per block
        at 480x832). Negligible against seconds of compute per block, but the cost is
        real and should not be reasoned about as if it were free.
        """
        session = self._get(session_id)
        stream = session.require_stream("session_drain")
        session.touch()

        chunks: list[torch.Tensor] = []
        for chunk in stream.blocks():
            chunks.append(chunk)
            session.blocks_emitted += 1
            if max_blocks is not None and len(chunks) >= max_blocks:
                break
        session.touch()
        logger.info(
            "[CF_SESSION] %s drained %d block(s), %d latents emitted total",
            session_id,
            len(chunks),
            stream.latents_emitted,
        )
        return chunks

    def session_decode_step(self, session_id: str, latents: torch.Tensor) -> torch.Tensor:
        """Decode one chunk of latents on a VAE-role session, seam-free.

        This is the stage-1 counterpart to ``session_drain``. It keeps the decoder's
        temporal ``feat_cache`` alive across calls, so the first frame of this chunk
        attends to the context the previous chunk left behind — the output is identical
        to decoding the whole clip in one pass, which is the property that makes
        chunk-by-chunk streaming visually seamless rather than merely convenient.
        """
        session = self._get(session_id)
        if session.stream is not None:
            raise CausalForcingSessionError(
                f"session_decode_step() is for VAE-role sessions; {session_id!r} owns a "
                f"rollout (role={session.role}) and decodes inline via session_drain()."
            )
        pipeline = self._require_role("session_decode_step", "vae", "full")
        session.touch()
        # Timed so that the pure device cost can be compared against the caller's
        # round trip (see [CF_BLOCK_TIME] in stage_worker). The two differing by
        # much means the time is going to the RPC path, not to the cards, and that
        # distinction has already been got wrong once here.
        t0 = time.monotonic()
        video = pipeline._decode_stream_step(latents)
        decode_ms = (time.monotonic() - t0) * 1e3
        session.chunks_decoded += 1
        session.touch()
        logger.info(
            "[CF_DECODE_TIME] %s chunk=%d decode=%.0fms latents=%s out=%s",
            session_id,
            session.chunks_decoded,
            decode_ms,
            tuple(latents.shape) if hasattr(latents, "shape") else "?",
            tuple(video.shape) if hasattr(video, "shape") else "?",
        )
        return video

    def session_close(self, session_id: str, *, missing_ok: bool = False) -> dict[str, Any]:
        """Release a session and its caches."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            if missing_ok:
                return {"session_id": session_id, "closed": False, "reason": "not found"}
            raise CausalForcingSessionError(f"Cannot close unknown session {session_id!r}.")
        stats = session.stats()
        self._teardown(session)
        logger.info("[CF_SESSION] closed %s: %s", session_id, stats)
        return {"session_id": session_id, "closed": True, **stats}

    def session_info(self, session_id: str | None = None) -> dict[str, Any]:
        """Report one session, or all of them plus worker capacity."""
        if session_id is not None:
            return self._get(session_id).stats()
        return {
            "role": self._stage_role(),
            "open_sessions": len(self._sessions),
            "sessions": [s.stats() for s in self._sessions.values()],
        }
