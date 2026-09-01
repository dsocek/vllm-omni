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

Aggregated (one diffusion stage): the whole pipeline -- text encode, framewise
rollout, and VAE decode -- in a single worker on a single device. Select it via
the deploy YAML ``pipeline: causal_forcing_agg`` key.

Under vllm-omni's own offline entrypoint an aggregated run needs no registry
entry: it falls back to ``_create_default_diffusion_stage_cfg`` (keyed off
model_index ``_class_name=CausalForcingPipeline``). That fallback lives in
``async_omni_engine.py`` and is reached only when the caller passes
``default_stage_cfg_factory``, which Dynamo's resolver
(``dynamo.vllm.omni.utils.resolve_stage_configs_compat``) does not. So under
Dynamo a single-stage diffusion deploy resolves to ZERO stages and every worker
dies with ``--stage-id 0 out of range (YAML has 0 stages)``.
CAUSAL_FORCING_AGG_PIPELINE below closes that gap, which is what makes an
aggregated-vs-disaggregated comparison possible on the same serving stack.
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

CAUSAL_FORCING_AGG_PIPELINE = PipelineConfig(
    model_type="causal_forcing_agg",
    # Same reasoning as above: no diffusers_class_name, so this never
    # auto-matches on _class_name="CausalForcingPipeline". Both Causal-Forcing
    # topologies are opt-in via the deploy YAML ``pipeline:`` key, which keeps
    # the two explicit and symmetric -- picking a topology by accident is what
    # would make a throughput comparison between them meaningless.
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            # The only stage, so it is the final one: it emits pixels, not
            # latents, and takes the "image" post-process func registered for
            # CausalForcingPipeline (the same one the disaggregated VAE stage
            # uses) rather than the DiT stage's latent passthrough.
            final_output=True,
            final_output_type="image",
            # _stage_role="full" on this class is what colocates rollout and
            # decode: CausalForcingPipeline.forward decodes each block inside
            # the rollout loop, so per-block cost is T_dit + T_vae.
            model_arch="CausalForcingPipeline",
        ),
    ),
)
