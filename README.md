# Talk to a video — V-JEPA2 + Whisper + Llama 3.2 on WebGPU

Browser-only demo: pick a video, get visual action predictions + a speech transcript, then chat about the video with a local Llama 3.2 1B. Everything runs on WebGPU. Nothing leaves the browser.

**Live page:** https://anentrypoint.github.io/vjepa2/ (once Pages is enabled)

## Models

| Role   | Model | Size | Source |
|--------|-------|------|--------|
| Vision | V-JEPA2 ViT-L finetuned on Something-Something V2 (174 generic action templates), fp16 ONNX, **16 frames** | ~720 MB | This repo's GH release `model-fp16-ssv2-v1` (CI exports [`facebook/vjepa2-vitl-fpc16-256-ssv2`](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) → ONNX → fp16) |
| ASR    | Whisper base, q8 ONNX | ~150 MB | [`Xenova/whisper-base`](https://huggingface.co/Xenova/whisper-base) via `@huggingface/transformers` |
| LLM    | Llama 3.2 1B Instruct, q4f16_1 | ~700 MB | [`mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC`](https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC) via `@mlc-ai/web-llm` |

All three run concurrently as Web Workers on a single WebGPU device. After the first load, weights are cached in IndexedDB.

## How "talk to a video" actually works

V-JEPA2 is a video encoder, not a multimodal LLM. There is no trained projector from V-JEPA2's 1024-dim features into Llama's embedding space — that would require a real training run on video-instruction data.

So we keep this honest and browser-feasible by **bridging through text**:

1. V-JEPA2 (SSV2 head) classifies 16 uniformly sampled frames, producing top-5 class predictions with probabilities. SSV2's 174 categories are generic action templates ("moving X up", "putting X into Y", "pretending to pick up X") that compose well over arbitrary handheld/everyday video — not just one domain.
2. Whisper transcribes the audio track.
3. Both are injected as a single `system` message into Llama 3.2 1B.
4. The user chats; Llama answers grounded in those observations.

SSV2 was picked over Diving48 for breadth (174 generic actions vs. 48 dive-specific labels) and over UCF101 / Kinetics finetunes because it's the smallest variant we have a clean export for and runs at 16 frames, halving activation VRAM and inference time vs. fpc32.

## VRAM

- Vision fp16: ~720 MB resident
- Whisper q8: ~150 MB
- Llama q4f16: ~700 MB
- Total: ~1.6 GB GPU; works on 4 GB+ GPUs.

fp16 was chosen for V-JEPA2 over int8/q4 because static quantization on a video ViT classification head can lose 1–3% top-1 without calibration data, and the user constraint was "no function loss." fp16 is effectively lossless for ViT inference.

## Run locally

```
node serve.js
# open http://localhost:8787
```

Requires Chrome/Edge with WebGPU (or Firefox with `dom.webgpu.enabled`, Safari TP).

## Files

| Path | Purpose |
|---|---|
| `public/index.html` | UI: video upload, three workers, chat |
| `public/worker-vision.js` | V-JEPA2 ONNX inference (onnxruntime-web + WebGPU) |
| `public/worker-asr.js` | Whisper via `@huggingface/transformers` |
| `public/worker-llm.js` | Llama 3.2 1B via `@mlc-ai/web-llm` |
| `serve.js` | Minimal Node static server with COOP/COEP |
| `scripts/convert_fp16.py` | fp32 → fp16 ONNX conversion |
| `.github/workflows/build-fp16.yml` | CI conversion + release publish |
| `test.js` | Sanity asserts |

## Vision inputs / outputs

- Input: `[1, 16, 3, 256, 256]` fp16, ImageNet-normalized, 16 frames sampled uniformly across video duration.
- Output: `[1, 174]` logits over SSV2 classes; softmax + top-5 for the chat context.
