---
title: VQwen QFormer Sludge Detector
emoji: 🥴
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
license: apache-2.0
models:
  - alpharomercoma/vqwen-qformer-tiktok
short_description: Detect internet brain-rot "sludge" videos from a frame.
---

# 🥴 VQwen-QFormer Sludge Detector

Gradio UI around [`alpharomercoma/vqwen-qformer-tiktok`](https://huggingface.co/alpharomercoma/vqwen-qformer-tiktok) — a BLIP-2 vision tower + Qwen3-4B LLM (LoRA merged) fine-tuned to identify TikTok-style **sludge**: videos stacking multiple unrelated feeds at once (the "brain-rot" layout).

## How it works

1. Upload a short video.
2. The app samples 1–3 frames uniformly across the clip.
3. Each frame is passed to the model with the prompt *"Is this sludge content? Answer yes or no."*
4. A majority vote across frames produces the final verdict.
5. Optional deep analysis adds per-frame **layout** and **description** outputs.

## Notes on the CPU Space

The model is 5B parameters. On the default free CPU hardware, a single yes/no generation takes tens of seconds. Defaults are tuned to stay responsive:

- One frame, one short generation per run.
- `do_sample=False`, `max_new_tokens=6` for the classification prompt.
- Weights load in `bfloat16` (≈10 GB) via `low_cpu_mem_usage=True`.

For snappier demos, duplicate the Space onto a GPU tier.
