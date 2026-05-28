"""Gradio Space for alpharomercoma/vqwen-qformer-tiktok-v2.

Takes a video, transcribes its audio with Whisper, samples a frame, and
classifies it as sludge vs non-sludge using the VQwen-QFormer v2 model
(BLIP-2 vision + Q-Former + frozen Linear projector + Qwen3-4B LoRA, with
the audio transcript concatenated into the prompt).

CPU-friendly: weights load in bfloat16; generation uses greedy decoding
with a tiny token budget for the yes/no prompt; Whisper is lazy-loaded on
first call to keep Space cold-start under HF's health-check budget;
deeper analysis (layout + description) is opt-in.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import gradio as gr
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration, pipeline

MODEL_ID = "alpharomercoma/vqwen-qformer-tiktok-v2"
ASR_MODEL_ID = "openai/whisper-base"  # 74M params, ~290 MB; CPU-friendly.
MAX_FRAMES = 3
MAX_TRANSCRIPT_CHARS = 600  # mirrors training-time truncation in vqwen-qformer/scripts/12_build_tiktok_convs_v2.py

PROMPT_CLASSIFY = "Is this sludge content? Answer yes or no."
PROMPT_LAYOUT = "What layout type is shown?"
PROMPT_DESCRIBE = "Describe this frame."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # matches the weights on disk; keeps memory ~10 GB

# Shipped alongside the model as chat_template.jinja / processor_config.json,
# but older Blip2Processor doesn't auto-populate self.chat_template on load,
# so we inline it here as a fallback. 32 <image> tokens = Q-Former query count.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\n' -}}"
    "{%- if message['content'] is string -%}"
    "{{- message['content'] -}}"
    "{%- else -%}"
    "{%- for item in message['content'] -%}"
    "{%- if item['type'] == 'image' -%}"
    + ("<image>" * 32) + "\n"
    "{%- elif item['type'] == 'text' -%}"
    "{{- item['text'] -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\n' -}}"
    "{%- endif -%}"
)


def _load():
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=DTYPE, low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if not getattr(processor, "chat_template", None):
        processor.chat_template = CHAT_TEMPLATE
    return model, processor


print(f"[startup] loading {MODEL_ID} on {DEVICE} ({DTYPE})...")
MODEL, PROCESSOR = _load()
print("[startup] model ready")


# Whisper is lazy-loaded on first call so the Space's cold-start stays under
# Hugging Face's ~30s health-check budget. Cost: first analyze() pays the
# ~10s ASR weight load once; subsequent calls reuse the loaded pipeline.
_ASR = None


def _get_asr():
    global _ASR
    if _ASR is None:
        print(f"[asr] loading {ASR_MODEL_ID} (first call only)...")
        _ASR = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_ID,
            device=-1,  # CPU
        )
        print("[asr] ready")
    return _ASR


def _extract_audio(video_path: str) -> Path | None:
    """Dump 16 kHz mono WAV to a temp file. Returns None if the video has no
    audio stream (silent video — not an error, just transcribe-as-empty)."""
    out = Path(tempfile.mkstemp(suffix=".wav")[1])
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
         "-f", "wav", str(out)],
        capture_output=True,
    )
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        return None
    return out


def _truncate(t: str, n: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Char-level, whitespace-aware truncation with ellipsis suffix.
    Mirrors vqwen-qformer/scripts/12_build_tiktok_convs_v2.py:122-126 exactly
    so inference prompt shape matches what the model saw during training."""
    if len(t) <= n:
        return t
    cut = t.rfind(" ", 0, n)
    return (t[:cut] if cut > 0 else t[:n]).rstrip() + "…"


def _get_transcript(video_path: str) -> str:
    """End-to-end: video -> audio -> Whisper -> truncated text. Returns ""
    on any failure (silent video, transcription crash, etc.); empty
    transcript triggers vision-only fallback in _ask, matching training-time
    behavior for transcript-less samples."""
    wav = _extract_audio(video_path)
    if wav is None:
        return ""
    try:
        result = _get_asr()(str(wav))
        text = (result.get("text") if isinstance(result, dict) else "") or ""
        return _truncate(text.strip())
    except Exception as e:  # noqa: BLE001
        print(f"[asr] transcription failed: {e}")
        return ""
    finally:
        try:
            wav.unlink()
        except OSError:
            pass


@dataclass
class FrameResult:
    index: int
    timestamp: float
    image: Image.Image
    verdict: str
    is_sludge: bool
    layout: str | None
    description: str | None


def _ask(image: Image.Image, question: str, max_new_tokens: int, transcript: str = "") -> str:
    # Build the user turn: when a transcript is present, prepend an
    # "Audio transcript: <text>" text block BEFORE the image, then the
    # question. This matches the training-time prompt layout from
    # vqwen-qformer/scripts/12_build_tiktok_convs_v2.py (the v2 model was
    # trained on prompts shaped this way; vision-only inference produces
    # measurably worse predictions).
    content: list[dict] = []
    if transcript:
        content.append({"type": "text", "text": f"Audio transcript: {transcript}"})
    content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    prompt = PROCESSOR.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = PROCESSOR(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
    with torch.inference_mode():
        out = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return PROCESSOR.decode(gen, skip_special_tokens=True).strip()


def _pick_indices(total: int, n_frames: int) -> List[int]:
    n = max(1, min(n_frames, MAX_FRAMES, total))
    if n == 1:
        return [total // 2]
    return [int(i) for i in np.linspace(0, total - 1, n, dtype=int)]


def _read_imageio(video_path: str, n_frames: int) -> List[Tuple[int, float, Image.Image]]:
    """Primary reader: imageio + PyAV (ffmpeg-backed, wide codec support)."""
    fps = 30.0
    n_total = 0
    for plugin in ("pyav", "FFMPEG", None):
        try:
            meta = iio.immeta(video_path, plugin=plugin) if plugin else iio.immeta(video_path)
            fps = float(meta.get("fps") or meta.get("fps ") or 30.0)
            duration = float(meta.get("duration") or 0.0)
            n_total = int(duration * fps) if duration > 0 else 0
            break
        except Exception:
            continue

    # Stream-count if metadata is unreliable.
    if n_total <= 0:
        all_frames = list(iio.imiter(video_path))
        n_total = len(all_frames)
        if n_total == 0:
            return []
        indices = _pick_indices(n_total, n_frames)
        return [
            (i, i / fps, Image.fromarray(all_frames[i]))
            for i in indices
        ]

    indices = _pick_indices(n_total, n_frames)
    out: List[Tuple[int, float, Image.Image]] = []
    for idx in indices:
        arr = None
        for plugin in ("pyav", "FFMPEG", None):
            try:
                arr = iio.imread(video_path, index=idx, plugin=plugin) if plugin else iio.imread(video_path, index=idx)
                break
            except Exception:
                continue
        if arr is None:
            continue
        out.append((idx, idx / fps, Image.fromarray(arr)))
    return out


def _read_cv2(video_path: str, n_frames: int) -> List[Tuple[int, float, Image.Image]]:
    """Fallback reader: OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        cap.release()
        return []
    indices = _pick_indices(total, n_frames)
    out: List[Tuple[int, float, Image.Image]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append((int(idx), float(idx) / fps, Image.fromarray(frame)))
    cap.release()
    return out


def _sample_frames(video_path: str, n_frames: int) -> List[Tuple[int, float, Image.Image]]:
    errors = []
    for reader in (_read_imageio, _read_cv2):
        try:
            out = reader(video_path, n_frames)
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            errors.append(f"{reader.__name__}: {e}")
    detail = " | ".join(errors) if errors else "no frames decoded"
    raise gr.Error(
        f"Couldn't decode the video ({detail}). Try re-encoding to MP4 (H.264 + AAC)."
    )


def analyze(video_path: str | None, n_frames: int, deep: bool, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload a video first.")

    progress(0.02, desc="Transcribing audio")
    transcript = _get_transcript(video_path)

    progress(0.10, desc="Sampling frames")
    samples = _sample_frames(video_path, int(n_frames))

    results: List[FrameResult] = []
    total_steps = len(samples) * (3 if deep else 1)
    step = 0

    for idx, ts, img in samples:
        step += 1
        progress(step / (total_steps + 1), desc=f"Classifying frame {len(results) + 1}/{len(samples)}")
        verdict = _ask(img, PROMPT_CLASSIFY, max_new_tokens=6, transcript=transcript)
        is_sludge = verdict.lower().lstrip().startswith("yes")

        layout = description = None
        if deep:
            step += 1
            progress(step / (total_steps + 1), desc=f"Layout for frame {len(results) + 1}")
            layout = _ask(img, PROMPT_LAYOUT, max_new_tokens=16, transcript=transcript)
            step += 1
            progress(step / (total_steps + 1), desc=f"Describing frame {len(results) + 1}")
            description = _ask(img, PROMPT_DESCRIBE, max_new_tokens=96, transcript=transcript)

        results.append(
            FrameResult(
                index=idx,
                timestamp=ts,
                image=img,
                verdict=verdict,
                is_sludge=is_sludge,
                layout=layout,
                description=description,
            )
        )

    sludge_votes = sum(1 for r in results if r.is_sludge)
    share = sludge_votes / len(results)
    is_sludge_overall = sludge_votes > len(results) / 2
    confidence = share if is_sludge_overall else 1.0 - share

    label = "SLUDGE" if is_sludge_overall else "NON-SLUDGE"
    emoji = "🥴" if is_sludge_overall else "✅"
    color = "#ff4b6e" if is_sludge_overall else "#1fbf75"
    sub = (
        "Multi-pane, unrelated content stacked together — classic brain-rot layout."
        if is_sludge_overall
        else "Single coherent scene — no sludge layout detected."
    )

    verdict_html = f"""
    <div class="verdict-card" style="--accent:{color}">
      <div class="verdict-emoji">{emoji}</div>
      <div class="verdict-body">
        <div class="verdict-label">{label}</div>
        <div class="verdict-sub">{sub}</div>
        <div class="verdict-meta">
          <span><b>{sludge_votes}</b> / {len(results)} frames flagged</span>
          <span>Agreement: <b>{confidence * 100:.0f}%</b></span>
        </div>
      </div>
    </div>
    """

    gallery = [
        (r.image, f"t={r.timestamp:.1f}s · {'SLUDGE' if r.is_sludge else 'non-sludge'} · {r.verdict}")
        for r in results
    ]

    table_rows = [
        [
            f"{r.timestamp:.2f}s",
            "🥴 sludge" if r.is_sludge else "✅ non-sludge",
            r.verdict,
            r.layout or "—",
            r.description or "—",
        ]
        for r in results
    ]

    transcript_display = (
        transcript if transcript
        else "(No speech detected by Whisper — the verdict above is vision-only.)"
    )

    return verdict_html, gallery, transcript_display, table_rows


CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto; }
#hero {
  padding: 28px 32px;
  border-radius: 20px;
  background: linear-gradient(135deg, #ec4899 0%, #a855f7 55%, #6366f1 100%);
  color: #ffffff !important;
  margin-bottom: 18px;
  box-shadow: 0 10px 30px -12px rgba(168, 85, 247, 0.45);
}
#hero h1 { margin: 0 0 8px 0; font-size: 30px; letter-spacing: -0.01em; color: #ffffff !important; }
#hero p  { margin: 0; font-size: 14px; color: rgba(255,255,255,0.9) !important; }
#hero code {
  background: rgba(255,255,255,0.18);
  color: #ffffff !important;
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 13px;
}

.verdict-card {
  display: flex; align-items: center; gap: 18px;
  padding: 22px 26px; border-radius: 16px;
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.08);
  border-left: 6px solid var(--accent);
  color: #111827 !important;
  box-shadow: 0 4px 18px -8px rgba(0,0,0,0.15);
}
.verdict-emoji { font-size: 54px; line-height: 1; }
.verdict-label {
  font-size: 26px; font-weight: 700; letter-spacing: 0.02em;
  color: var(--accent) !important;
}
.verdict-sub { color: #374151 !important; margin-top: 4px; font-size: 14px; }
.verdict-meta {
  display: flex; gap: 20px; margin-top: 10px; font-size: 13px; color: #6b7280 !important;
}
.verdict-meta b { color: #111827; }

@media (prefers-color-scheme: dark) {
  .verdict-card {
    background: #1f2937;
    border-color: rgba(255,255,255,0.08);
    color: #f9fafb !important;
  }
  .verdict-sub { color: #d1d5db !important; }
  .verdict-meta { color: #9ca3af !important; }
  .verdict-meta b { color: #f9fafb; }
}
"""

THEME = gr.themes.Soft(primary_hue="pink", neutral_hue="slate")

with gr.Blocks(title="VQwen-QFormer · Sludge Detector") as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>🥴 Sludge Detector</h1>
          <p>Upload a short video. We transcribe the audio with Whisper, sample
          a frame, and ask <code>vqwen-qformer-tiktok-v2</code> whether the
          clip is internet brain-rot sludge — multiple unrelated streams
          playing at once — or a single coherent scene.
          <br><b>Expect ~3-5 minutes per video on the free CPU tier.</b></p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Video", sources=["upload"], height=320)
            n_frames = gr.Slider(
                minimum=1, maximum=MAX_FRAMES, step=1, value=1,
                label="Frames to sample",
                info="More frames = more reliable vote, but each frame adds ~30-90s on CPU.",
            )
            deep = gr.Checkbox(
                label="Deep analysis (layout + description per frame)",
                value=False,
                info="Adds two extra generations per frame. Expect many minutes on CPU.",
            )
            run_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            verdict = gr.HTML(label="Verdict")
            gallery = gr.Gallery(
                label="Sampled frames",
                columns=3, height=220, object_fit="cover", show_label=True,
            )
            transcript_out = gr.Textbox(
                label="Audio transcript (Whisper)",
                placeholder="Whisper transcript will appear here after analysis.",
                lines=4, max_lines=10, show_copy_button=True, interactive=False,
            )

    with gr.Accordion("Per-frame details", open=False):
        table = gr.Dataframe(
            headers=["Timestamp", "Classification", "Model output", "Layout", "Description"],
            datatype=["str"] * 5,
            wrap=True,
            interactive=False,
        )

    gr.Markdown(
        f"""
        <small>
        Model: <a href="https://huggingface.co/{MODEL_ID}" target="_blank"><code>{MODEL_ID}</code></a>
        · EVA-CLIP-G/14 + Q-Former + frozen Linear projector + Qwen3-4B (LoRA merged) ·
        Audio transcription: <a href="https://huggingface.co/{ASR_MODEL_ID}" target="_blank"><code>{ASR_MODEL_ID}</code></a> ·
        Running on <b>{DEVICE.upper()}</b> in {str(DTYPE).split('.')[-1]}.
        End-to-end on free CPU: ~3-5 min per video (audio + 1 frame × ~30-90s of generation).
        For snappier inference, <a href="https://huggingface.co/spaces/alpharomercoma/vqwen-qformer/discussions" target="_blank">duplicate this Space</a> and select <b>ZeroGPU</b> hardware.
        </small>
        """
    )

    run_btn.click(
        fn=analyze,
        inputs=[video, n_frames, deep],
        outputs=[verdict, gallery, transcript_out, table],
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(theme=THEME, css=CSS)
