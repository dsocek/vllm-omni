# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flat parallel kwargs must reach the nested ``parallel_config``.

``OmniDiffusionConfig`` declares parallelism only as a nested ``parallel_config``,
and ``from_kwargs`` filters to declared fields. A stage YAML that sets a flat
``tensor_parallel_size: 2`` was therefore silently dropped, and the stage came up
with an all-ones parallel config -- the knob appeared to be set and did nothing.
CLI overrides were unaffected because ``_apply_diffusion_parallel_runtime_overrides``
already nests them; only YAML-declared flat keys were lost.
"""

import pytest

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestFlatParallelKwargs:
    def test_flat_tensor_parallel_size_is_folded(self):
        config = OmniDiffusionConfig.from_kwargs(model="test", tensor_parallel_size=2)
        assert config.parallel_config.tensor_parallel_size == 2
        assert config.parallel_config.world_size == 2

    def test_flat_vae_patch_parallel_is_folded(self):
        config = OmniDiffusionConfig.from_kwargs(
            model="test",
            vae_patch_parallel_size=2,
            vae_parallel_mode="tile",
        )
        assert config.parallel_config.vae_patch_parallel_size == 2
        assert config.parallel_config.vae_parallel_mode == "tile"

    def test_vae_patch_parallel_does_not_grow_world_size(self):
        """Patch parallelism reuses the DiT group rather than growing it.

        This is why a VAE stage needs its own ``tensor_parallel_size`` to get a
        multi-rank group: ``_get_world_rank_pp_size`` clamps to
        ``min(vae_patch_parallel_size, world_size)``.
        """
        config = OmniDiffusionConfig.from_kwargs(
            model="test",
            tensor_parallel_size=1,
            vae_patch_parallel_size=2,
        )
        assert config.parallel_config.world_size == 1

        config = OmniDiffusionConfig.from_kwargs(
            model="test",
            tensor_parallel_size=2,
            vae_patch_parallel_size=2,
        )
        assert config.parallel_config.world_size == 2

    def test_nested_parallel_config_wins_over_flat(self):
        config = OmniDiffusionConfig.from_kwargs(
            model="test",
            tensor_parallel_size=2,
            parallel_config={"tensor_parallel_size": 4},
        )
        assert config.parallel_config.tensor_parallel_size == 4

    def test_flat_keys_fill_gaps_in_nested_config(self):
        config = OmniDiffusionConfig.from_kwargs(
            model="test",
            vae_patch_parallel_size=2,
            parallel_config={"tensor_parallel_size": 2},
        )
        assert config.parallel_config.tensor_parallel_size == 2
        assert config.parallel_config.vae_patch_parallel_size == 2

    def test_prebuilt_parallel_config_object_is_untouched(self):
        parallel_config = DiffusionParallelConfig(tensor_parallel_size=4)
        config = OmniDiffusionConfig.from_kwargs(
            model="test",
            tensor_parallel_size=2,
            parallel_config=parallel_config,
            num_gpus=4,
        )
        assert config.parallel_config.tensor_parallel_size == 4

    def test_no_flat_keys_leaves_defaults(self):
        config = OmniDiffusionConfig.from_kwargs(model="test")
        assert config.parallel_config.tensor_parallel_size == 1
        assert config.parallel_config.vae_patch_parallel_size == 1
        assert config.parallel_config.world_size == 1

    def test_none_valued_flat_keys_are_ignored(self):
        """Stage configs carry ``tensor_parallel_size: None`` for unset knobs."""
        config = OmniDiffusionConfig.from_kwargs(
            model="test",
            tensor_parallel_size=None,
            vae_patch_parallel_size=None,
        )
        assert config.parallel_config.tensor_parallel_size == 1
        assert config.parallel_config.vae_patch_parallel_size == 1
