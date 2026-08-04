# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seed-determinism tests for CausalForcingStream.

These run the real ``blocks()`` loop -- the queue walk, the transition handling, and
the generator selection -- against a stub pipeline whose ``_rollout_block`` records
the noise it was handed instead of denoising it. That is the whole point: what a seed
controls is the *noise draw*, so recording the draws is a stronger check than
comparing output tensors, and it needs no model and no device.

The property under test is the one a storyboard depends on. Re-rolling one shot with
``push_scene(seed=...)`` must leave every *other* scene bit-identical, which is only
true because a per-scene seed gets its own generator rather than reseeding the
stream's shared sequence in place.
"""

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.causal_forcing.pipeline_causal_forcing import (
    CausalForcingPipeline,
    CausalForcingStream,
)

LATENT_FRAMES = 1  # nfpb: one block per latent keeps the recorded draws easy to index


class StubTransformer:
    """Just the attributes CausalForcingStream reads off the transformer."""

    num_layers = 1
    dim = 8
    num_heads = 2
    patch_size = (1, 2, 2)
    in_dim = 4
    local_attn_size = 21
    text_len = 512

    def __init__(self) -> None:
        self.blocks = [type("B", (), {"self_attn": type("A", (), {})()})()]


class StubPipe:
    """Minimal stand-in for CausalForcingPipeline.

    Real methods are borrowed where they are pure bookkeeping (``_init_caches``);
    ``_rollout_block`` is replaced by a recorder, and ``encode_prompt`` by a constant,
    so no weights are needed.
    """

    _stage_role = "dit"
    denoising_step_list = torch.tensor([1000, 750])
    denoising_step_list_first_chunk = torch.tensor([1000, 750, 500])

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.num_frame_per_block = LATENT_FRAMES
        self.transformer = StubTransformer()
        self.vae = None  # DiT role: blocks() yields latents, no decode
        self.text_encoder = torch.nn.Linear(1, 1)  # only .parameters() is touched
        self.noise_draws: list[torch.Tensor] = []
        # Which denoising schedule each block got. Only the first block of a shot may
        # get the first-chunk one; see the resume suite.
        self.denoise_lists: list[torch.Tensor] = []
        # Each block's position within its shot. This is where a cut is visible without
        # a real attention implementation: a cut rebases it to 0, so re-applying one
        # mid-scene shows up here as the position going backwards.
        self.current_starts: list[int] = []

    # Borrowed unchanged: allocating the caches is pure bookkeeping, and using the
    # real one keeps the stream's cache layout honest rather than mocked around.
    _init_caches = CausalForcingPipeline._init_caches

    def encode_prompt(self, prompt, max_sequence_length=None):
        return torch.zeros(1, 4, 8)

    def _rollout_block(self, *, noisy_block, generator, denoise_list=None, current_start=0, **kwargs):
        # Record the draw, not the denoise: the seed's only job is to fix this tensor.
        self.noise_draws.append(noisy_block.clone())
        self.denoise_lists.append(denoise_list)
        self.current_starts.append(current_start)
        return noisy_block


def _run(scenes) -> list[torch.Tensor]:
    """Play a storyboard and return the per-block noise draws.

    ``scenes`` is a list of ``(prompt, kwargs)``; kwargs go straight to push_scene.
    """
    pipe = StubPipe()
    stream = CausalForcingStream(pipe, height=16, width=16, seed=1234)
    for prompt, kwargs in scenes:
        stream.push_scene(prompt, latents=LATENT_FRAMES, **kwargs)
    list(stream.blocks())
    stream.close()
    return pipe.noise_draws


BOARD = [
    ("scene one", {"transition": "cut"}),
    ("scene two", {}),
    ("scene three", {}),
]


def test_stream_seed_makes_the_whole_board_reproducible():
    """open_stream(seed=...) replays a storyboard identically end to end."""
    first, second = _run(BOARD), _run(BOARD)
    assert len(first) == len(BOARD)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_unseeded_stream_is_not_reproducible():
    """Guards the test above from passing vacuously (e.g. if noise were constant)."""
    pipes = []
    for _ in range(2):
        pipe = StubPipe()
        stream = CausalForcingStream(pipe, height=16, width=16)  # no seed
        stream.push_scene("scene one", latents=LATENT_FRAMES, transition="cut")
        list(stream.blocks())
        stream.close()
        pipes.append(pipe)
    assert not torch.equal(pipes[0].noise_draws[0], pipes[1].noise_draws[0])


def test_scene_seed_is_reproducible_on_its_own():
    """The same scene seed reproduces that scene's noise, run to run."""
    board = [("scene one", {"transition": "cut", "seed": 77})]
    assert torch.equal(_run(board)[0], _run(board)[0])


def test_scene_seed_overrides_the_stream_seed_for_that_scene_only():
    """The storyboard property: re-rolling one shot must not disturb the others.

    A per-scene generator is what buys this. Reseeding the stream's generator in place
    would rewind the shared sequence, so scenes two and three would silently change
    as well -- a re-roll of shot one would quietly re-render the rest of the board.
    """
    baseline = _run(BOARD)

    rerolled = list(BOARD)
    rerolled[0] = ("scene one", {"transition": "cut", "seed": 999})
    got = _run(rerolled)

    assert not torch.equal(got[0], baseline[0]), "seeded scene should have new noise"
    for i in (1, 2):
        assert torch.equal(got[i], baseline[i]), f"scene {i} must be untouched"


def test_two_scene_seeds_are_independent():
    """Seeding a later scene leaves earlier ones alone too."""
    baseline = _run(BOARD)
    board = list(BOARD)
    board[2] = ("scene three", {"seed": 5})
    got = _run(board)
    assert torch.equal(got[0], baseline[0])
    assert torch.equal(got[1], baseline[1])
    assert not torch.equal(got[2], baseline[2])


def test_same_scene_seed_on_different_scenes_draws_the_same_noise():
    """Confirms the override really is the generator, with nothing else mixed in."""
    board = [
        ("scene one", {"transition": "cut", "seed": 42}),
        ("scene two", {"seed": 42}),
    ]
    draws = _run(board)
    assert torch.equal(draws[0], draws[1])


def test_scene_seed_zero_is_honoured():
    """0 is a real seed; a falsy check anywhere on the path would drop it."""
    board_zero = [("scene one", {"transition": "cut", "seed": 0})]
    assert torch.equal(_run(board_zero)[0], _run(board_zero)[0])
    # ...and it is not the same as leaving the seed unset (which uses the stream's).
    board_unset = [("scene one", {"transition": "cut"})]
    assert not torch.equal(_run(board_zero)[0], _run(board_unset)[0])


def test_push_scene_rejects_a_non_int_seed():
    pipe = StubPipe()
    stream = CausalForcingStream(pipe, height=16, width=16)
    try:
        stream.push_scene("scene", latents=LATENT_FRAMES, seed="42")
    except ValueError as e:
        assert "seed must be an int" in str(e)
    else:
        raise AssertionError("a string seed should be rejected, not coerced")
    stream.close()
