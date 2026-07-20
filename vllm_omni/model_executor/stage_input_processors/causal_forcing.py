# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for disaggregated Causal-Forcing: DiT -> VAE.

Stage 0 (``CausalForcingDiTPipeline``) runs the framewise rollout and emits
raw model-space latents. This bridge pulls that latent tensor out of the DiT
stage's ``OmniRequestOutput`` and packages it as the prompt for stage 1
(``CausalForcingVAEPipeline``), which decodes it to pixels.

Modeled on ``stage_input_processors.glm_image.ar2diffusion`` (which passes
``prior_token_ids`` through ``prompt['extra']``); here the payload is a video
latent tensor instead of tokens.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _latent_from_output(source_output: Any) -> torch.Tensor | None:
    """Extract the DiT-stage latent tensor from its stage output.

    Primary (disaggregated / dynamo) channel: the DiT post-process routes the
    latent onto ``multimodal_output['latent']`` of the completion output, which
    is the payload the inter-stage connector preserves. So read
    ``source_output.outputs[0].multimodal_output['latent']`` first. Fall back to
    the in-process channels (``custom_output['latents']`` / ``.images[0]``) that
    the single-host offline path produces, for robustness.
    """
    if source_output is None:
        return None

    # Primary: multimodal_output.latent on the completion output(s).
    outputs = getattr(source_output, "outputs", None)
    if outputs:
        for o in outputs:
            mm = getattr(o, "multimodal_output", None)
            if isinstance(mm, dict) and mm.get("latent") is not None:
                return mm["latent"]

    # Fallbacks for the in-process offline path.
    custom_output = getattr(source_output, "custom_output", None)
    if isinstance(custom_output, dict) and custom_output.get("latents") is not None:
        return custom_output["latents"]

    mm = getattr(source_output, "multimodal_output", None)
    if isinstance(mm, dict) and mm.get("latent") is not None:
        return mm["latent"]

    images = getattr(source_output, "images", None)
    if images:
        first = images[0] if isinstance(images, list) else images
        if isinstance(first, torch.Tensor):
            return first

    inner = getattr(source_output, "request_output", None)
    if inner is not None and inner is not source_output:
        return _latent_from_output(inner)

    return None


def dit2vae(
    source_outputs: list[Any],
    prompt: Any | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> dict[str, Any] | None:
    """Turn DiT-stage output into the VAE stage's input prompt dict.

    Returns ``{"prompt", "height", "width", "num_frames", "output_type",
    "extra": {"latents": <tensor>}}`` or ``None`` if no latent is available
    (the orchestrator routes a terminal error in that case).
    """
    del requires_multimodal_data, streaming_context

    _t0 = time.perf_counter()
    if not source_outputs:
        return None

    latents = _latent_from_output(source_outputs[0])
    if latents is None:
        logger.warning("[dit2vae] no latent tensor found on DiT-stage output; skipping request")
        return None
    if not isinstance(latents, torch.Tensor):
        latents = torch.as_tensor(latents)

    # Original request prompt (carry through user-facing generation params).
    if isinstance(prompt, list):
        original_prompt = prompt[0] if prompt else {}
    else:
        original_prompt = prompt if prompt is not None else {}
    if not isinstance(original_prompt, dict):
        if hasattr(original_prompt, "_asdict"):
            original_prompt = original_prompt._asdict()
        elif hasattr(original_prompt, "__dict__"):
            original_prompt = vars(original_prompt)
        else:
            original_prompt = {}

    text_prompt = original_prompt.get("prompt", "")
    # Latent geometry -> pixel geometry (VAE upsamples 8x spatial, 4x temporal).
    # latents: [B, C, latent_frames, latent_h, latent_w]
    latent_frames = int(latents.shape[2])
    latent_h = int(latents.shape[3])
    latent_w = int(latents.shape[4])
    height = original_prompt.get("height") or latent_h * 8
    width = original_prompt.get("width") or latent_w * 8
    num_frames = original_prompt.get("num_frames") or (latent_frames - 1) * 4 + 1

    vae_input: dict[str, Any] = {
        "prompt": text_prompt,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        # Ensure the VAE stage decodes to pixels (never re-enters a latent path).
        "output_type": original_prompt.get("output_type", "np"),
        "extra": {"latents": latents},
    }

    logger.info(
        "[dit2vae] latents=%s -> target=%dx%d frames=%d wall=%.3fms",
        tuple(latents.shape),
        vae_input["width"],
        vae_input["height"],
        vae_input["num_frames"],
        (time.perf_counter() - _t0) * 1000.0,
    )
    return vae_input
