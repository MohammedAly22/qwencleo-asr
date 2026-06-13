"""QwenCleo-ASR Gradio demo.

Run (after installing requirements / `pip install qwencleo-asr[app]`):

    python app/gradio_app.py            # mic + file upload UI on :7860
    QWENCLEO_MODEL=... python app/gradio_app.py
"""
from __future__ import annotations

import os

import gradio as gr

from qwencleo_asr import QwenCleoASR
from qwencleo_asr.streaming import stream_file

MODEL_ID = os.environ.get("QWENCLEO_MODEL", "mohammedaly22/QwenCleo-ASR")
DEVICE = os.environ.get("QWENCLEO_DEVICE", "cuda:0")

asr = QwenCleoASR(MODEL_ID, device=DEVICE)

GOLD = "#D4AF37"


def transcribe_fn(audio_path, language, do_normalize):
    if not audio_path:
        return "Please provide audio."
    lang = None if language == "Auto-detect" else language
    r = asr.transcribe(audio_path, language=lang, normalize=do_normalize)
    return r.text


def stream_fn(audio_path, language):
    """Stream a long file chunk by chunk, updating the textbox live."""
    if not audio_path:
        yield "Please provide audio."
        return
    lang = None if language == "Auto-detect" else language
    acc = []
    for chunk in stream_file(asr, audio_path, language=lang):
        acc.append(chunk.text)
        yield " ".join(acc)


with gr.Blocks(
    title="QwenCleo-ASR",
    theme=gr.themes.Soft(primary_hue=gr.themes.colors.yellow),
) as demo:
    gr.Markdown(
        f"""
        # 🏺 QwenCleo-ASR
        Egyptian Arabic & code-switching speech recognition — built on Qwen3-ASR.
        <span style="color:{GOLD}">**Tuned for Egyptian dialect + Arabic/English code-switching.**</span>
        """
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                 label="Audio (mic or file)")
            language = gr.Dropdown(
                ["Arabic", "Auto-detect", "English"], value="Arabic",
                label="Language hint",
            )
            do_norm = gr.Checkbox(label="Egyptian normalization", value=False)
            with gr.Row():
                btn = gr.Button("Transcribe", variant="primary")
                btn_stream = gr.Button("Stream (long audio)")
        with gr.Column():
            out = gr.Textbox(label="Transcript", lines=10, rtl=True)

    btn.click(transcribe_fn, [audio_in, language, do_norm], out)
    btn_stream.click(stream_fn, [audio_in, language], out)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
