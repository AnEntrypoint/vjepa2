# V-JEPA2 ViT-L (diving48) — WebGPU demo

Browser-only video-classification demo. Runs Meta's V-JEPA2 ViT-L finetuned on Diving48 directly on the GPU via `onnxruntime-web` + WebGPU.

**Live page:** https://anentrypoint.github.io/vjepa2/ (after Pages enabled)
**Source model:** [`onnx-community/vjepa2-vitl-fpc32-256-diving48-ONNX`](https://huggingface.co/onnx-community/vjepa2-vitl-fpc32-256-diving48-ONNX)

## Why this variant

Of the 86 V-JEPA2 models on HF, this is the only one with a published ONNX export, which means zero conversion friction for `onnxruntime-web`. ViT-L (~0.4B params) is the smallest tier with an ONNX build — ViT-H (0.7B) and ViT-g (1B) would blow past WebGPU buffer limits.

## VRAM strategy: fp16

The published ONNX is fp32 (~1.4&nbsp;GB) which exceeds typical WebGPU single-buffer caps. We convert it once, in CI, to fp16 with [`onnxconverter-common`](https://github.com/microsoft/onnxconverter-common):

- **~720 MB on disk and resident** (half of fp32).
- **Effectively lossless** for ViT inference vs fp32 — fp16 is the standard inference dtype for vision transformers; we did not pick int8/q4 because static quantization for video transformers requires calibration data and can shave 1–3% top-1 on classification heads.

The conversion runs in a GitHub Actions workflow ([.github/workflows/build-fp16.yml](.github/workflows/build-fp16.yml)) and publishes `model.fp16.onnx` as a release asset on the `model-fp16-v1` tag. The browser fetches the release asset directly.

## Run locally

```
node serve.js
# open http://localhost:8787
```

Requires a browser with WebGPU (Chrome/Edge, Safari TP, Firefox with `dom.webgpu.enabled`).

## Build the fp16 weights yourself

```
py -3 -m pip install onnx onnxconverter-common huggingface_hub hf_transfer
hf download onnx-community/vjepa2-vitl-fpc32-256-diving48-ONNX onnx/model.onnx --local-dir models_dl
mkdir -p models && mv models_dl/onnx/model.onnx models/model.onnx
py -3 scripts/convert_fp16.py
```

## Files

| Path | Purpose |
|---|---|
| `public/index.html` | Single-page WebGPU inference (frame extraction + ORT Web + softmax + top-5) |
| `serve.js` | Minimal Node static server with COOP/COEP headers |
| `scripts/convert_fp16.py` | fp32 → fp16 ONNX conversion |
| `.github/workflows/build-fp16.yml` | CI conversion + release publish |
| `test.js` | Minimal integration test |

## Inputs / outputs

- Input: `[1, 32, 3, 256, 256]` fp16, ImageNet-normalized (mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`), 32 frames sampled uniformly across the video duration.
- Output: `[1, 48]` logits over Diving48 classes (id2label in `config.json`).
