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
  - alpharomercoma/vqwen-qformer-tiktok-v2
  - openai/whisper-base
short_description: Detect TikTok-style "sludge" videos (multi-pane brain-rot)
---

# 🥴 VQwen-QFormer Sludge Detector

Gradio UI around [`alpharomercoma/vqwen-qformer-tiktok-v2`](https://huggingface.co/alpharomercoma/vqwen-qformer-tiktok-v2) — a tri-modal classifier built on EVA-CLIP-G/14 + Q-Former + Linear projector (frozen in stage 2) + Qwen3-4B (LoRA merged), with a Whisper transcript of the clip's audio concatenated into the prompt at inference time. Fine-tuned to identify TikTok-style **sludge**: videos stacking multiple unrelated feeds at once (the "brain-rot" layout).

## How it works

1. Upload a short video.
2. The audio is extracted with `ffmpeg` and transcribed by [`openai/whisper-base`](https://huggingface.co/openai/whisper-base).
3. One to three frames are sampled uniformly across the clip.
4. Each frame is passed to the model with the prompt `Audio transcript: <text>\n<image>\nIs this sludge content? Answer yes or no.` and the model emits a yes/no.
5. A majority vote across frames produces the final verdict.
6. Optional deep analysis adds per-frame **layout** and **description** outputs (each grounded in the audio transcript).

## Notes on the CPU Space

The model is 4B parameters and Whisper-base is another 74M. On the default free CPU hardware, end-to-end takes ~3-5 minutes per video. Defaults are tuned to stay responsive:

- One frame and one short generation per run by default.
- Whisper is **lazy-loaded** on first `Analyze` click so Space cold-start stays under HF's health-check budget.
- `do_sample=False`, `max_new_tokens=6` for the classification prompt.
- Weights load in `bfloat16` (≈10 GB) via `low_cpu_mem_usage=True`.
- The audio transcript is truncated char-by-char to 600 characters — same as the training-time truncation in the canonical pipeline.

For snappier demos: duplicate this Space and select **ZeroGPU** hardware. A100 access on demand drops total inference to ~30-60s per video. (Requires HF Pro for the developer.)

## Known caveat: Whisper-size skew

The v2 model was trained on transcripts produced by Whisper-V3-Turbo (809M params, ~1.5 GB). This Space uses the much smaller `whisper-base` (74M) to keep CPU latency tolerable. For clean speech the difference is small; for music-heavy or non-speech audio the smaller model is more prone to hallucinated repetition, which can dent prediction quality vs. the headline benchmark numbers. If you swap to a GPU tier, consider also upgrading to `openai/whisper-large-v3-turbo` to match training conditions.
