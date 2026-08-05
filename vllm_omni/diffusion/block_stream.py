# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Picklable, lazily-polled stream of latent blocks for a disaggregated VAE decode.

A disaggregated video pipeline can decode pixels *while* the upstream DiT stage is
still rolling out latents, instead of waiting for the whole clip. The producer
writes each latent block to shared memory under the per-chunk key
``{request_id}_c{n}`` and, once the last block is written, a stream-end marker
under ``{request_id}_cend``.

The consumer side is what this module provides. The decode runs in a **diffusion
worker process**, not in the serving worker that talks to the connector: the
executor pickles the request through ``shm_broadcast`` to reach it. So the block
source cannot be a queue, a socket, a connector instance, or anything else holding
a lock or a file handle — any of those raise ``TypeError: cannot pickle
'_thread.lock' object`` at broadcast time. It must be a **plain description of
where to look**, which reconstitutes itself after unpickling and does its own
shared-memory reads in the decode process.

That is ``ShmLatentBlockStream``: cheap to pickle (a request id, a couple of stage
labels, a config dict, and the first block's tensor), and iterable exactly once in
the process that decodes. Iterating blocks until each next block appears, and stops
at the stream-end marker — so a caller's ``for block in stream`` loop needs no
count, no async, and no knowledge of any of this.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterator

from vllm.logger import init_logger

logger = init_logger(__name__)

# Poll cadence. The shared-memory read is non-blocking, so a consumer running
# concurrently with the producer has to poll. Back off from _MIN to _MAX: a fast
# producer is picked up promptly, while a long rollout does not spin a core. A
# latent block takes O(100ms)+ to produce, so _MAX bounds the added latency per
# block at a small fraction of that.
POLL_MIN_S = 0.002
POLL_MAX_S = 0.05

# Stall diagnostics. A producer that dies *without* writing the end marker looks
# exactly like a slow one from here, and would otherwise poll forever — a silent
# hang, the most expensive failure to diagnose on a remote run. So warn while
# waiting, then fail with a specific error.
#
# Two separate bounds, because absence means different things at different points:
# _FIRST covers block c0, whose wait legitimately includes upstream model warmup;
# _GAP covers a later block, where the producer has already proven it is alive and
# emitting, so a long gap means it died mid-rollout. Bounding the gap tightly is
# what turns a multi-minute mystery stall into a prompt, specific error.
#
# Public because the producer-side worker waits for block c0 on the async event
# loop before it can hand this stream over, and that wait must use the same
# cadence and bounds as the in-process polling below.
STALL_WARN_S = 5.0
FIRST_TIMEOUT_S = float(os.environ.get("DYN_CMAF_FIRST_BLOCK_TIMEOUT_S", "600"))
GAP_TIMEOUT_S = float(os.environ.get("DYN_CMAF_BLOCK_GAP_TIMEOUT_S", "120"))


def chunk_key(request_id: str, chunk_index: int) -> str:
    """Per-block shared-memory key: ``{request_id}_c{n}``.

    Derived from the request id and index alone, so producer and consumer agree on
    the address with no handshake.
    """
    return f"{request_id}_c{chunk_index}"


def chunk_end_key(request_id: str) -> str:
    """Stream-end marker key: ``{request_id}_cend``.

    Outside the ``_c{int}`` namespace, so it can never collide with a block key.
    """
    return f"{request_id}_cend"


def latent_from_payload(payload: Any) -> Any:
    """Pull the latent tensor out of a connector payload.

    Mirrors the serving-side reader: the producer's post-process routes the latent
    onto ``multimodal_output['latent']``, but *which* multimodal_output survives the
    connector round-trip varies, so all three channels the producer may have used
    are checked — the request-level one, a completion output's, and the sidecar list
    the producer writes precisely because some serializers drop completion-only
    fields. Returns None when the payload carries no latent.
    """
    import torch

    if payload is None:
        return None

    sidecar: Any = None
    # Producer may wrap engine_inputs alongside preserved completion attributes.
    if isinstance(payload, dict):
        if payload.get("stream_end"):
            return None
        sidecar = payload.get("_dynamo_completion_output_attrs")
        payload = payload.get("engine_inputs", payload)

    def _latent(mm: Any) -> Any:
        if isinstance(mm, dict) and isinstance(mm.get("latent"), torch.Tensor):
            return mm["latent"]
        return None

    if (latent := _latent(getattr(payload, "multimodal_output", None))) is not None:
        return latent
    for output in getattr(payload, "outputs", None) or []:
        if (latent := _latent(getattr(output, "multimodal_output", None))) is not None:
            return latent
    # Sidecar last: it is a copy, so prefer the live objects when they survived.
    for attrs in sidecar or []:
        if isinstance(attrs, dict):
            if (latent := _latent(attrs.get("multimodal_output"))) is not None:
                return latent
    images = getattr(payload, "images", None)
    if images:
        first = images[0] if isinstance(images, (list, tuple)) else images
        if isinstance(first, torch.Tensor):
            return first
    return None


class ShmLatentBlockStream:
    """Latent blocks read from shared memory as the producer writes them.

    Picklable by construction — it holds only data, never a lock, handle, or
    connection — so it survives the executor's broadcast to the decode process,
    where ``__iter__`` does the actual reading.

    ``first_block`` is the already-fetched block ``c0``, kept as a plain tensor so
    a consumer can read geometry (``[B, C, *, H, W]``) *without* consuming the
    stream; iteration still yields it first, so the decode sees the full sequence.
    """

    def __init__(
        self,
        request_id: str,
        from_stage: Any,
        to_stage: Any,
        connector_config: dict[str, Any] | None,
        first_block: Any,
    ) -> None:
        self.request_id = request_id
        self.from_stage = str(from_stage)
        self.to_stage = str(to_stage)
        # Config only (a plain dict from YAML) — never a connector instance.
        self.connector_config = dict(connector_config or {})
        self.first_block = first_block

    # -- consumer side (runs in the decode process) --------------------------

    def _make_reader(self) -> Any:
        """Build a fresh shared-memory reader in *this* process.

        The connector is deliberately reconstructed here rather than shipped: it is
        a stateless, config-only object, so rebuilding it costs nothing and reuses
        the connector's own locking and partial-write handling instead of
        re-implementing shared-memory reads (and their races) here.
        """
        from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
            SharedMemoryConnector,
        )

        return SharedMemoryConnector(self.connector_config)

    def _try_read(self, reader: Any, key: str) -> Any | None:
        """One non-blocking read: payload, or None if the key is not there yet."""
        try:
            result = reader.get(self.from_stage, self.to_stage, key)
        except Exception:
            # A miss can surface as a raise; treat it as "not present yet".
            return None
        if result is None:
            return None
        # `get` returns (obj, size) on the SHM path.
        payload = result[0] if isinstance(result, tuple) else result
        if payload is None:
            return None
        # A reader can observe a segment mid-write and deserialize an empty
        # container; that is "not ready yet", not a block.
        if isinstance(payload, (list, tuple, dict, str, bytes)) and len(payload) == 0:
            return None
        return payload

    def _await_block(self, reader: Any, n: int) -> Any | None:
        """Poll for block ``n``; return its payload, or None once the stream ends.

        The end marker is checked *after* a failed block read, never before: it is
        written only once every block is in shared memory, so marker-present plus
        block-absent proves block ``n`` will never exist. Checking in that order
        (and re-reading the block once) closes the race where the block and the
        marker both land between the two reads.
        """
        delay = POLL_MIN_S
        t0 = time.monotonic()
        next_warn = STALL_WARN_S
        polls = 0
        timeout = FIRST_TIMEOUT_S if n == 0 else GAP_TIMEOUT_S

        while True:
            payload = self._try_read(reader, chunk_key(self.request_id, n))
            if payload is not None:
                waited = time.monotonic() - t0
                if waited >= STALL_WARN_S:
                    logger.info(
                        "[cmaf-step2] block c%d of %s arrived after %.1fs (%d polls)",
                        n, self.request_id, waited, polls,
                    )
                return payload

            end = self._try_read(reader, chunk_end_key(self.request_id))
            if end is not None:
                payload = self._try_read(reader, chunk_key(self.request_id, n))
                if payload is not None:
                    logger.info(
                        "[cmaf-step2] block c%d of %s landed in the end-marker race "
                        "window (recovered, not a lost block)",
                        n, self.request_id,
                    )
                    return payload
                expected = end.get("num_chunks") if isinstance(end, dict) else None
                logger.info(
                    "[cmaf-step2] stream end confirmed for %s at c%d (producer "
                    "reported %s blocks)",
                    self.request_id, n, expected,
                )
                if expected is not None and int(expected) != n:
                    # The marker's count is authoritative: a mismatch means blocks
                    # were lost or overwritten, not a clean end.
                    logger.error(
                        "[cmaf-step2] BLOCK COUNT MISMATCH for %s — producer wrote "
                        "%s blocks but the feed stopped at c%d; the decoded video "
                        "will be truncated",
                        self.request_id, expected, n,
                    )
                return None

            waited = time.monotonic() - t0
            if waited >= next_warn:
                logger.warning(
                    "[cmaf-step2] still waiting for block c%d of %s after %.1fs "
                    "(%d polls, no stream-end marker). Upstream is either slow or "
                    "died without publishing the marker; will fail at %.0fs.",
                    n, self.request_id, waited, polls, timeout,
                )
                next_warn += STALL_WARN_S
            if waited >= timeout:
                raise RuntimeError(
                    f"timed out after {waited:.0f}s waiting for block c{n} of "
                    f"{self.request_id} with no stream-end marker (upstream stage "
                    f"{self.from_stage} "
                    + (
                        "never produced a block — check that it started and is "
                        "block-streaming; raise DYN_CMAF_FIRST_BLOCK_TIMEOUT_S if "
                        "warmup is simply this slow"
                        if n == 0
                        else f"emitted c0..c{n - 1} then stopped, so it likely died "
                        "mid-rollout; raise DYN_CMAF_BLOCK_GAP_TIMEOUT_S if a single "
                        "block legitimately takes this long"
                    )
                    + ")"
                )
            polls += 1
            time.sleep(delay)
            delay = min(delay * 2, POLL_MAX_S)

    def __iter__(self) -> Iterator[Any]:
        """Yield latent blocks in order, blocking until each one is available.

        Block ``c0`` is yielded from ``first_block`` (already in hand, so no read),
        then ``c1..`` are polled from shared memory. Stops at the stream-end marker.
        """
        # [cmaf-trace] Iteration starts in the *decode* process, after unpickling.
        # Pairs with the worker-side handover log: if that one appears and this one
        # does not, the descriptor never reached the decode (or was never iterated).
        logger.info(
            "[cmaf-trace] pid=%d block_stream: __iter__ ENTERED for %s "
            "(first_block %s)",
            os.getpid(),
            self.request_id,
            tuple(self.first_block.shape)
            if hasattr(self.first_block, "shape")
            else type(self.first_block).__name__,
        )
        yield self.first_block

        reader = self._make_reader()
        n = 1
        t_start = time.monotonic()
        while True:
            payload = self._await_block(reader, n)
            if payload is None:
                logger.info(
                    "[cmaf-step2] block stream for %s exhausted after %d blocks (%.2fs)",
                    self.request_id, n, time.monotonic() - t_start,
                )
                return
            latent = latent_from_payload(payload)
            if latent is None:
                # Truncating here would silently shorten the video, so say so.
                logger.error(
                    "[cmaf-step2] block c%d of %s carried no latent tensor; stopping "
                    "at %d blocks — the decoded video will be truncated",
                    n, self.request_id, n,
                )
                return
            logger.info(
                "[cmaf-step2] block c%d %s of %s handed to decode (+%.2fs)",
                n, tuple(latent.shape), self.request_id, time.monotonic() - t_start,
            )
            yield latent
            n += 1
