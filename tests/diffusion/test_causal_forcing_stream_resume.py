# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mid-scene resume tests for ``CausalForcingStream.blocks()``.

The pipelined service drains with ``max_blocks=1``: it takes one block, breaks out of
the generator, ships the block downstream, and calls ``session_drain`` again -- which
builds a *fresh* generator. So every guarantee here is about state that has to live on
the queued scene rather than in the generator frame that just got thrown away.

This is a regression suite. ``blocks()`` used to ``popleft()`` the scene on entry, so a
one-block drain carried the scene off with the discarded generator: a 25-latent scene
emitted block 0 and then reported an empty queue, silently dropping the other 24. The
aggregated path never saw it, because it drains with ``max_blocks=None`` and finishes
inside one generator's lifetime.

Nothing here would have caught it through a fake stream. The session-extension suite
already asserted the right contract (``test_drain_respects_max_blocks``) against a
``FakeStream`` whose queue is an integer counter -- which survives generator
abandonment for free, so the fake passed while the real loop lost the scene. These
tests drive the real generator, reusing the recording stub from the seed suite.
"""

from __future__ import annotations

from test_causal_forcing_stream_seed import LATENT_FRAMES, StubPipe

from vllm_omni.diffusion.models.causal_forcing.pipeline_causal_forcing import (
    CausalForcingStream,
)


def _stream(seed: int | None = 1234) -> tuple[StubPipe, CausalForcingStream]:
    pipe = StubPipe()
    return pipe, CausalForcingStream(pipe, height=16, width=16, seed=seed)


def _drain(stream, max_blocks=None) -> list:
    """One ``session_drain`` call: take up to ``max_blocks``, then drop the generator.

    Deliberately a faithful copy of ``session_extension.session_drain``'s loop,
    including that the generator is abandoned rather than closed, because the abandoning
    is what these tests are about.
    """
    out = []
    for block in stream.blocks():
        out.append(block)
        if max_blocks is not None and len(out) >= max_blocks:
            break
    return out


# -- the scene survives a one-block drain -----------------------------------


def test_a_one_block_drain_leaves_the_rest_of_the_scene_queued():
    """The bug, stated directly: 25 latents must not become 1."""
    _, stream = _stream()
    stream.push_scene("a long shot", latents=25, transition="cut")

    assert len(_drain(stream, max_blocks=1)) == 1
    assert stream.pending_latents == 24, "the undrained blocks must still be queued"


def test_draining_one_block_at_a_time_yields_every_block():
    """What the pipelined route actually does, for a whole scene."""
    _, stream = _stream()
    stream.push_scene("a long shot", latents=25, transition="cut")

    blocks = []
    while True:
        got = _drain(stream, max_blocks=1)
        if not got:
            break
        blocks.extend(got)

    assert len(blocks) == 25
    assert stream.latents_emitted == 25 * LATENT_FRAMES
    assert stream.pending_latents == 0
    assert stream.scenes_played == 1, "the scene should be counted once, when it ended"


def test_one_at_a_time_matches_a_single_full_drain_block_for_block():
    """Chunking the drain must not change the video.

    The strong form of the property: the noise draws -- and so the output -- have to be
    identical whether the caller took one block per call or all of them at once. This is
    what pins the scene's generator to the scene: a fresh generator reseeded on re-entry
    would redraw block 0's noise for block 1, and the clip would stutter rather than
    fail.
    """
    pipe_chunked, chunked = _stream()
    chunked.push_scene("a long shot", latents=6, transition="cut")
    while _drain(chunked, max_blocks=1):
        pass

    pipe_whole, whole = _stream()
    whole.push_scene("a long shot", latents=6, transition="cut")
    _drain(whole)

    assert len(pipe_chunked.noise_draws) == len(pipe_whole.noise_draws) == 6
    for i, (a, b) in enumerate(zip(pipe_chunked.noise_draws, pipe_whole.noise_draws, strict=True)):
        assert a.equal(b), f"block {i} drew different noise when the drain was chunked"


def test_a_resumed_scene_does_not_redraw_the_same_noise():
    """Guards the test above from passing vacuously if every draw were equal anyway."""
    pipe, stream = _stream()
    stream.push_scene("a long shot", latents=3, transition="cut")
    while _drain(stream, max_blocks=1):
        pass

    first, second, third = pipe.noise_draws
    assert not first.equal(second)
    assert not second.equal(third)


# -- the once-per-scene transitions stay once-per-scene ----------------------


def test_only_the_scenes_first_block_uses_the_first_chunk_schedule():
    """A resumed middle block must not be denoised as if it opened the shot.

    ``denoising_step_list_first_chunk`` exists because a cut's first block has no
    history to attend to. Re-applying it to block 3 of the same scene would denoise a
    block that does have history as though it did not.
    """
    pipe, stream = _stream()
    stream.push_scene("a long shot", latents=4, transition="cut")
    while _drain(stream, max_blocks=1):
        pass

    schedules = pipe.denoise_lists
    assert schedules[0] is pipe.denoising_step_list_first_chunk
    for i, sched in enumerate(schedules[1:], start=1):
        assert sched is pipe.denoising_step_list, f"block {i} reused the first-chunk schedule"


def test_a_resumed_cut_does_not_rebase_the_shot_again():
    """The cut's history reset happens once, on the scene's first block.

    Running it again mid-scene would re-zero ``scene_base_frame`` against the *current*
    cursor, so the shot's position would restart from 0 part way through and the blocks
    already in the window would fall outside it -- the shot restarting from noise with
    nothing reporting an error.

    Observed through ``current_start``, the position each block is rolled out at, since
    the recording stub does not write to the KV cache and so cannot move its indices.
    """
    pipe, stream = _stream()
    stream.push_scene("first shot", latents=2, transition="cut")
    _drain(stream)
    starts_before = len(pipe.current_starts)

    # A second cut: its own first block legitimately rebases to 0, and every block after
    # it must keep climbing from there.
    stream.push_scene("second shot", latents=4, transition="cut")
    while _drain(stream, max_blocks=1):
        pass

    shot = pipe.current_starts[starts_before:]
    assert shot[0] == 0, "the cut's first block should open the shot at position 0"
    assert shot == sorted(shot) and len(set(shot)) == len(shot), (
        f"positions went backwards or repeated within the shot: {shot}"
    )


# -- scene boundaries -------------------------------------------------------


def test_a_drain_that_ends_on_a_scene_boundary_moves_to_the_next_scene():
    """An exactly-exhausting drain must not leave the finished scene at the head."""
    _, stream = _stream()
    stream.push_scene("first shot", latents=2, transition="cut")
    stream.push_scene("second shot", latents=2)

    assert len(_drain(stream, max_blocks=2)) == 2
    assert stream.scenes_played == 1
    assert stream.pending_latents == 2

    assert len(_drain(stream)) == 2, "the second scene must still roll out"
    assert stream.scenes_played == 2
    assert stream.pending_latents == 0


def test_a_scene_pushed_mid_scene_is_played_after_it():
    """Pushes and drains interleave, which is the point of a session."""
    _, stream = _stream()
    stream.push_scene("first shot", latents=3, transition="cut")
    _drain(stream, max_blocks=1)
    stream.push_scene("second shot", latents=2)

    assert stream.pending_latents == 2 + 2 * LATENT_FRAMES
    remaining = []
    while True:
        got = _drain(stream, max_blocks=1)
        if not got:
            break
        remaining.extend(got)
    assert len(remaining) == 4, "2 left of the first scene plus 2 of the second"
    assert stream.scenes_played == 2


def test_pending_latents_counts_undrained_blocks_only():
    """What a service reads for backpressure.

    Summing each scene's original ``latents`` would report a stream draining one block
    at a time as permanently full, and backpressure would never release.
    """
    _, stream = _stream()
    stream.push_scene("a long shot", latents=5, transition="cut")
    assert stream.pending_latents == 5
    for expected in (4, 3, 2, 1, 0):
        _drain(stream, max_blocks=1)
        assert stream.pending_latents == expected


def test_closing_mid_scene_drops_the_rest():
    """close() clears the queue, so a partly-drained scene does not resume after it."""
    _, stream = _stream()
    stream.push_scene("a long shot", latents=5, transition="cut")
    _drain(stream, max_blocks=1)
    stream.close()

    assert stream.pending_latents == 0
