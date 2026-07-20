# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline DISAGGREGATED Causal-Forcing text-to-video (2 diffusion stages).

Stage 0 (DiT) runs on one card and emits latents; stage 1 (VAE) runs on a
second card and decodes them. Uses the offline ``Omni`` multi-stage path with
``deploy_config=causal_forcing_disagg.yaml`` — no server required.

Device pinning: per-stage ``devices:`` in the YAML are LOGICAL indices into the
visible set, so launch masked to exactly the two reserved cards, e.g.

    ZE_AFFINITY_MASK=0,2 VLLM_XPU_ENABLE_XPU_GRAPH=0 HF_HOME=/mnt/bigtmp/hf \
    python examples/offline_inference/causal_forcing/disagg_t2v.py \
      --model /mnt/bigtmp/causal-forcing-1step \
      --prompt "A serene lakeside sunrise with mist over the water." \
      --num-frames 81 --height 480 --width 832 --fps 16 --seed 42 \
      --output out_disagg.mp4

Stage 0 -> physical card 0, stage 1 -> physical card 2. No other card is used.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

_DEFAULT_DEPLOY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "vllm_omni",
    "deploy",
    "causal_forcing_disagg.yaml",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Disaggregated (2-stage) Causal-Forcing text-to-video.")
    p.add_argument("--model", required=True, help="Assembled causal-forcing model dir.")
    p.add_argument("--deploy-config", default=None, help="Deploy YAML (default: causal_forcing_disagg.yaml).")
    p.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--output", type=str, default="causal_forcing_disagg.mp4")
    p.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="Submit N identical requests at once to measure cross-stage pipelining throughput.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    deploy_config = args.deploy_config or _DEFAULT_DEPLOY

    omni = Omni(model=args.model, deploy_config=deploy_config, enforce_eager=True)
    print(f"[disagg] num_stages={omni.num_stages} deploy={deploy_config}")

    # Per-stage sampling params: stage 0 (DiT) carries the generation params;
    # stage 1 (VAE) gets geometry via the dit2vae prompt dict.
    params_list = list(omni.default_sampling_params_list)
    generator = torch.Generator(device="xpu").manual_seed(args.seed) if torch.xpu.is_available() else None
    for sp in params_list:
        if isinstance(sp, OmniDiffusionSamplingParams):
            sp.height = args.height
            sp.width = args.width
            sp.num_frames = args.num_frames
            sp.num_inference_steps = 1
            sp.seed = args.seed
            if generator is not None:
                sp.generator = generator

    prompt_dict = {"prompt": args.prompt, "height": args.height, "width": args.width, "num_frames": args.num_frames}

    if args.num_requests > 1:
        # Throughput mode: submit N requests. With disaggregation the orchestrator
        # pipelines stages — DiT rolls out request k+1 on card 0 while the VAE
        # decodes request k on card 2 — so wall time approaches
        # DiT + N*VAE (VAE-bound) rather than N*(DiT+VAE).
        prompts = [dict(prompt_dict) for _ in range(args.num_requests)]
        t0 = time.perf_counter()
        outputs = omni.generate(prompts, params_list)
        dt = time.perf_counter() - t0
        print(
            f"[disagg] [CF_THROUGHPUT] {args.num_requests} reqs in {dt:.4f} s "
            f"= {dt / args.num_requests:.4f} s/req ({args.num_requests / dt:.3f} req/s)",
            flush=True,
        )
        frames = _extract_video(outputs if not isinstance(outputs, list) else outputs[:1] or outputs)
        if frames is not None:
            _save_video(frames, args.output, args.fps)
            print(f"[disagg] Saved first generated video to {args.output}")
        return

    t0 = time.perf_counter()
    outputs = omni.generate(prompt_dict, params_list)
    dt = time.perf_counter() - t0
    print(f"[disagg] [CF_E2E] {dt:.4f} s ({dt * 1000:.2f} ms)", flush=True)

    # Pull the decoded video from the final (VAE) stage output.
    frames = _extract_video(outputs)
    if frames is None:
        raise ValueError("No video frames found in disaggregated output.")

    _save_video(frames, args.output, args.fps)
    print(f"[disagg] Saved generated video to {args.output}")


def _extract_video(outputs):
    """Find the decoded video tensor/array in the (possibly nested) outputs."""
    if isinstance(outputs, list):
        # Prefer the final-stage (image) output.
        for ro in outputs:
            if getattr(ro, "stage_id", None) == 1 or getattr(ro, "final_output_type", None) == "image":
                v = _video_from_ro(ro)
                if v is not None:
                    return v
        for ro in outputs:
            v = _video_from_ro(ro)
            if v is not None:
                return v
        return None
    return _video_from_ro(outputs)


def _video_from_ro(ro):
    if isinstance(ro, OmniRequestOutput):
        inner = getattr(ro, "request_output", None)
        if inner is not None and inner is not ro:
            v = _video_from_ro(inner)
            if v is not None:
                return v
        images = getattr(ro, "images", None)
        if images:
            first = images[0] if isinstance(images, list) else images
            if isinstance(first, dict):
                return first.get("frames") or first.get("video")
            return first
    if isinstance(ro, (torch.Tensor, np.ndarray, list)):
        return ro
    return None


def _save_video(frames, output_path: str, fps: int) -> None:
    from diffusers.utils import export_to_video

    if isinstance(frames, torch.Tensor):
        v = frames.detach().cpu()
        if v.dim() == 5:
            v = v[0].permute(1, 2, 3, 0) if v.shape[1] in (3, 4) else v[0]
        elif v.dim() == 4 and v.shape[0] in (3, 4):
            v = v.permute(1, 2, 3, 0)
        if v.is_floating_point():
            v = v.clamp(-1, 1) * 0.5 + 0.5
        video_array = v.float().numpy()
    elif isinstance(frames, np.ndarray):
        video_array = frames[0] if frames.ndim == 5 else frames
        if np.issubdtype(video_array.dtype, np.integer):
            video_array = video_array.astype(np.float32) / 255.0
    else:
        video_array = frames

    if isinstance(video_array, np.ndarray):
        if video_array.ndim == 5:
            video_array = list(video_array[0])
        elif video_array.ndim == 4:
            video_array = list(video_array)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video_array, str(out), fps=fps)


if __name__ == "__main__":
    main()
