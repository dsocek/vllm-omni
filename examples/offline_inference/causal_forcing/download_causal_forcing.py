# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Assemble a self-contained Causal-Forcing model directory for vllm-omni.

The Causal-Forcing checkpoint (``zhuhz22/Causal-Forcing``) ships only the
generator DiT weights. The VAE, UMT5 text encoder, and tokenizer come from the
Wan2.1-T2V-1.3B base (``Wan-AI/Wan2.1-T2V-1.3B``). This script downloads both
and writes a ``model_index.json`` so vllm-omni routes the directory to
``CausalForcingPipeline``.

Assembled layout:
    <output-dir>/
        model_index.json
        framewise-1step.pt          # transformer weights (generator_ema)
        tokenizer/                  # UMT5 tokenizer   (from Wan2.1-T2V-1.3B)
        text_encoder/               # UMT5EncoderModel (from Wan2.1-T2V-1.3B)
        vae/                        # AutoencoderKLWan (from Wan2.1-T2V-1.3B)

Usage:
    python download_causal_forcing.py --output-dir ./causal-forcing-1step
"""

import argparse
import json
import os
import shutil
import time

from huggingface_hub import hf_hub_download, snapshot_download

CF_REPO = "zhuhz22/Causal-Forcing"
CF_CHECKPOINT = "causal-forcing++/framewise-1step.pt"
# Diffusers-format base: provides vae/ (AutoencoderKLWan), text_encoder/
# (UMT5EncoderModel), and tokenizer/ in the subfolder layout the pipeline loads.
WAN_BASE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _timed_snapshot(repo_id: str, local_dir: str, allow_patterns: list[str]) -> None:
    print(f"Downloading {allow_patterns} from {repo_id} -> {local_dir}")
    start = time.time()
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    print(f"  finished in {time.time() - start:.1f}s")


def main(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 1. Transformer checkpoint (single .pt with generator_ema).
    ckpt_dst = os.path.join(output_dir, "framewise-1step.pt")
    if not os.path.exists(ckpt_dst):
        print(f"Downloading {CF_CHECKPOINT} from {CF_REPO}")
        start = time.time()
        cached = hf_hub_download(repo_id=CF_REPO, filename=CF_CHECKPOINT)
        shutil.copyfile(cached, ckpt_dst)
        print(f"  finished in {time.time() - start:.1f}s -> {ckpt_dst}")
    else:
        print(f"Transformer checkpoint already present: {ckpt_dst}")

    # 2. Base components from Wan2.1-T2V-1.3B (tokenizer / text_encoder / vae).
    _timed_snapshot(
        WAN_BASE_REPO,
        output_dir,
        allow_patterns=[
            "tokenizer/*",
            "text_encoder/*",
            "vae/*",
        ],
    )

    # 3. model_index.json — routes to CausalForcingPipeline and carries the
    #    framewise-1step inference schedule constants.
    model_index = {
        "_class_name": "CausalForcingPipeline",
        "transformer_checkpoint": "framewise-1step.pt",
        "num_frame_per_block": 1,
        "context_noise": 0,
        "timestep_shift": 5.0,
        "local_attn_size": -1,
        "denoising_step_list": [1000],
        "denoising_step_list_first_chunk": [1000, 750, 500, 250],
    }
    index_path = os.path.join(output_dir, "model_index.json")
    with open(index_path, "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"Wrote {index_path}")
    print(f"\nAssembled Causal-Forcing model at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble a Causal-Forcing model directory for vllm-omni.")
    parser.add_argument("--output-dir", default="./causal-forcing-1step", help="Destination model directory.")
    args = parser.parse_args()
    main(args.output_dir)
