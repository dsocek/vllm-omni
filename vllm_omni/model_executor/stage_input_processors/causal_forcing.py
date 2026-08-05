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
    # ``latents`` may be a single whole-clip tensor (aggregated / fetch-all) or a
    # list of per-block tensors (incremental feed, §4.2 Step 1). Carry whichever
    # shape through to the VAE stage untouched — the pipeline flattens and streams
    # the blocks one frame at a time — and derive pixel geometry from the first
    # block (all blocks share [B, C, *, H, W]; only the temporal extent differs).
    if (first_block := getattr(latents, "first_block", None)) is not None:
        # Live block stream (§4.2 Step 2): the upstream stage is still producing.
        # Iterating here would block until the rollout finished and defeat the
        # overlap, so pass the stream through untouched and take geometry from its
        # peekable first block — all blocks share [B, C, *, H, W]. The total frame
        # count is unknowable now; fall back to the request's own num_frames below
        # (the real decode derives geometry from the latents, not from this).
        geom = first_block
        latent_frames = 0
    elif isinstance(latents, (list, tuple)):
        blocks = [b if isinstance(b, torch.Tensor) else torch.as_tensor(b) for b in latents]
        if not blocks:
            logger.warning("[dit2vae] empty latent block list on DiT-stage output; skipping request")
            return None
        latents = blocks
        geom = blocks[0]
        latent_frames = sum(int(b.shape[2]) for b in blocks)
    else:
        if not isinstance(latents, torch.Tensor):
            latents = torch.as_tensor(latents)
        geom = latents
        latent_frames = int(latents.shape[2])

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
    # geom: [B, C, latent_frames, latent_h, latent_w] (first block when chunked).
    latent_h = int(geom.shape[3])
    latent_w = int(geom.shape[4])
    height = original_prompt.get("height") or latent_h * 8
    width = original_prompt.get("width") or latent_w * 8
    # latent_frames == 0 marks a live stream of unknown length: only the request's
    # own num_frames is meaningful then (the count-based fallback needs the total).
    num_frames = original_prompt.get("num_frames") or (
        (latent_frames - 1) * 4 + 1 if latent_frames else 0
    )

    vae_input: dict[str, Any] = {
        "prompt": text_prompt,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        # Ensure the VAE stage decodes to pixels (never re-enters a latent path).
        "output_type": original_prompt.get("output_type", "np"),
        "extra": {"latents": latents},
    }

    if first_block is not None:
        latents_desc = f"live-stream x{tuple(geom.shape)}"
        source_desc = "live block stream (concurrent with upstream rollout)"
    elif isinstance(latents, list):
        latents_desc = f"{len(latents)}x{tuple(geom.shape)}"
        source_desc = f"block-list ({len(latents)} blocks, no cat)"
    else:
        latents_desc = tuple(latents.shape)
        source_desc = "single tensor"
    logger.info("[cmaf-step2] dit2vae latent source: %s", source_desc)
    logger.info(
        "[dit2vae] latents=%s -> target=%dx%d frames=%d wall=%.3fms",
        latents_desc,
        vae_input["width"],
        vae_input["height"],
        vae_input["num_frames"],
        (time.perf_counter() - _t0) * 1000.0,
    )
    return vae_input
