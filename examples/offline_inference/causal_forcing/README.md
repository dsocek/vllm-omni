# Causal-Forcing (framewise-1step) — text-to-video

[thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) is a
Wan2.1-T2V-1.3B-based **causal / autoregressive few-step** video diffusion model.
Video latents are generated block-by-block with a KV cache so new frames attend
causally to already-generated history. The `framewise-1step` variant does a
single denoise step per latent frame → extremely low latency.

This example serves the `causal-forcing++/framewise-1step` checkpoint under the
base vLLM-Omni diffusion engine (the pipeline owns its KV cache; no AR-Diffusion
engine required).

## 1. Assemble the model directory

The published checkpoint contains only the generator DiT. The VAE + UMT5 text
encoder / tokenizer come from the Wan2.1-T2V-1.3B base, so the download script
assembles both into one directory with a `model_index.json`:

```bash
python examples/offline_inference/causal_forcing/download_causal_forcing.py \
    --output-dir ./causal-forcing-1step
```

Result:

```
causal-forcing-1step/
├── model_index.json          # routes to CausalForcingPipeline + schedule
├── framewise-1step.pt        # generator DiT (generator_ema)
├── tokenizer/                # from Wan-AI/Wan2.1-T2V-1.3B
├── text_encoder/             # UMT5EncoderModel
└── vae/                      # AutoencoderKLWan
```

## 2. Generate a video (offline)

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
    --model ./causal-forcing-1step \
    --model-class-name CausalForcingPipeline \
    --prompt "A serene lakeside sunrise with mist over the water." \
    --num-inference-steps 1 \
    --num-frames 81 --height 480 --width 832 --fps 16 \
    --enforce-eager \
    --output causal_forcing_output.mp4
```

Notes:
- `--num-inference-steps 1` selects the framewise-1step schedule (`[1000]` per
  block; the first chunk uses a 4-step schedule `[1000, 750, 500, 250]`
  internally). These come from `model_index.json`.
- The distilled few-step generator runs **without CFG** (`guidance_scale = 1.0`).
- Default resolution is 480×832, 81 frames — matching the reference config
  (latent shape `[1, 21, 16, 60, 104]`).
