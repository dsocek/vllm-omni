# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal-Forcing pipeline topologies (frozen).

Disaggregated (two diffusion stages):
  Stage 0: DiT — text encode + framewise rollout, emits video latents.
  Stage 1: VAE — decodes latents to pixels.

This is the first diffusion->diffusion stage split in vllm-omni (existing
splits are AR->diffusion). The latent tensor flows stage 0 -> stage 1 via the
``dit2vae`` stage input processor (analogous to GLM-Image's ``ar2diffusion``,
but carrying a latent tensor instead of tokens).

The aggregated (single-stage) Causal-Forcing run does NOT use this topology —
it stays on the ``_create_default_diffusion_stage_cfg`` fallback (model_index
``_class_name=CausalForcingPipeline``). Select this disaggregated topology via
the deploy YAML ``pipeline: causal_forcing_disagg`` key.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.causal_forcing"

CAUSAL_FORCING_DISAGG_PIPELINE = PipelineConfig(
    model_type="causal_forcing_disagg",
    # Intentionally NO diffusers_class_name: the model ships
    # _class_name="CausalForcingPipeline", and auto-matching on it would
    # hijack the single-stage aggregated run. This disaggregated topology is
    # opt-in ONLY via the deploy YAML ``pipeline: causal_forcing_disagg`` key.
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=False,
            # Producer of latents; no post-process func is registered for this
            # arch, so the raw latent tensor is carried downstream untouched.
            final_output_type="latent",
            engine_output_type="latent",
            model_arch="CausalForcingDiTPipeline",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="vae",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            model_arch="CausalForcingVAEPipeline",
            custom_process_input_func=f"{_PROC}.dit2vae",
        ),
    ),
)
