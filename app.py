"""Gradio Space for alpharomercoma/vqwen-qformer-tiktok.

Takes a video, samples frames, and classifies each as sludge vs non-sludge
using the VQwen-QFormer (BLIP-2 + Qwen3-4B) model. CPU-friendly: weights
load in bfloat16, generation uses greedy decoding with a tiny token budget
for the yes/no prompt, and deeper analysis is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import gradio as gr
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

MODEL_ID = "alpharomercoma/vqwen-qformer-tiktok"
MAX_FRAMES = 3

PROMPT_CLASSIFY = "Is this sludge content? Answer yes or no."
PROMPT_LAYOUT = "What layout type is shown?"
PROMPT_DESCRIBE = "Describe this frame."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # matches the weights on disk; keeps memory ~10 GB


def _load():
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=DTYPE, low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


print(f"[startup] loading {MODEL_ID} on {DEVICE} ({DTYPE})...")
MODEL, PROCESSOR = _load()
print("[startup] model ready")


@dataclass
class FrameResult:
    index: int
    timestamp: float
    image: Image.Image
    verdict: str
    is_sludge: bool
    layout: str | None
    description: str | None


def _ask(image: Image.Image, question: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
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

    progress(0.05, desc="Sampling frames")
    samples = _sample_frames(video_path, int(n_frames))

    results: List[FrameResult] = []
    total_steps = len(samples) * (3 if deep else 1)
    step = 0

    for idx, ts, img in samples:
        step += 1
        progress(step / (total_steps + 1), desc=f"Classifying frame {len(results) + 1}/{len(samples)}")
        verdict = _ask(img, PROMPT_CLASSIFY, max_new_tokens=6)
        is_sludge = verdict.lower().lstrip().startswith("yes")

        layout = description = None
        if deep:
            step += 1
            progress(step / (total_steps + 1), desc=f"Layout for frame {len(results) + 1}")
            layout = _ask(img, PROMPT_LAYOUT, max_new_tokens=16)
            step += 1
            progress(step / (total_steps + 1), desc=f"Describing frame {len(results) + 1}")
            description = _ask(img, PROMPT_DESCRIBE, max_new_tokens=96)

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

    return verdict_html, gallery, table_rows


CSS = """
:root { --bg: #0b0b10; }
.gradio-container { max-width: 1100px !important; margin: 0 auto; }
#hero {
  padding: 28px 32px;
  border-radius: 20px;
  background: radial-gradient(circle at 20% 0%, #ff4b6e33, transparent 60%),
              radial-gradient(circle at 100% 100%, #7c5cff33, transparent 55%),
              linear-gradient(135deg, #15151f, #0f0f16);
  color: #f5f5fa;
  margin-bottom: 18px;
  border: 1px solid #2a2a38;
}
#hero h1 { margin: 0 0 6px 0; font-size: 30px; letter-spacing: -0.01em; }
#hero p  { margin: 0; opacity: 0.75; font-size: 14px; }
.verdict-card {
  display: flex; align-items: center; gap: 18px;
  padding: 22px 26px; border-radius: 18px;
  background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 16%, #12121a), #11111a);
  border: 1px solid color-mix(in srgb, var(--accent) 40%, #2a2a38);
  color: #f5f5fa;
}
.verdict-emoji { font-size: 54px; line-height: 1; }
.verdict-label {
  font-size: 28px; font-weight: 700; letter-spacing: 0.02em;
  color: var(--accent);
}
.verdict-sub { opacity: 0.85; margin-top: 2px; font-size: 14px; }
.verdict-meta {
  display: flex; gap: 18px; margin-top: 10px; font-size: 13px; opacity: 0.85;
}
footer { display: none !important; }
"""

THEME = gr.themes.Soft(primary_hue="pink", neutral_hue="slate")

with gr.Blocks(title="VQwen-QFormer · Sludge Detector") as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>🥴 Sludge Detector</h1>
          <p>Upload a short video. We sample a few frames and ask
          <code>vqwen-qformer-tiktok</code> whether it's internet brain-rot sludge —
          multiple unrelated things playing at once — or a single coherent scene.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Video", sources=["upload"], height=320)
            n_frames = gr.Slider(
                minimum=1, maximum=MAX_FRAMES, step=1, value=1,
                label="Frames to sample",
                info="More frames = more reliable vote, but slower on CPU.",
            )
            deep = gr.Checkbox(
                label="Deep analysis (layout + description per frame)",
                value=False,
                info="Adds two extra generations per frame. Expect minutes on CPU.",
            )
            run_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            verdict = gr.HTML(label="Verdict")
            gallery = gr.Gallery(
                label="Sampled frames",
                columns=3, height=220, object_fit="cover", show_label=True,
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
        · BLIP-2 vision tower + Qwen3-4B LLM with LoRA merged ·
        Running on <b>{DEVICE.upper()}</b> in {str(DTYPE).split('.')[-1]}.
        CPU inference for a 5B model is slow — expect ~30–90s per frame.
        </small>
        """
    )

    run_btn.click(
        fn=analyze,
        inputs=[video, n_frames, deep],
        outputs=[verdict, gallery, table],
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(theme=THEME, css=CSS)
