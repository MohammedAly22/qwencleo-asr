#!/usr/bin/env python3
"""Generate the example Colab notebooks for qwencleo-asr.

Run:  python examples/_build_notebooks.py
Emits *.ipynb into the examples/ folder. Each notebook is cell-by-cell runnable
on Google Colab (GPU runtime) and self-contained (installs deps, fetches a
sample clip).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# A short Egyptian/code-switch sample hosted in the repo (raw URL). Notebooks
# download it so users have audio to try without uploading anything.
SAMPLE_URL = ("https://huggingface.co/mohammedaly22/QwenCleo-ASR/resolve/main/"
              "assets/sample.wav")
# Fallback: the public Qwen demo clip (English) if the repo sample is absent.
FALLBACK_URL = ("https://qianwen-res.oss-cn-beijing.aliyuncs.com/"
                "Qwen3-ASR-Repo/asr_en.wav")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def _src(lines):
    # join into a list of lines with trailing newlines except the last
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def write(name, nb):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", path)


# ---- shared cells ---------------------------------------------------------
BANNER = md(
    "# 🏺 QwenCleo-ASR — {title}",
    "",
    "Egyptian Arabic & code-switching speech recognition, built on Qwen3-ASR.",
    "",
    "[Model](https://huggingface.co/mohammedaly22/QwenCleo-ASR) · "
    "[GitHub](https://github.com/MohammedAly22/qwencleo-asr) · "
    "[PyPI](https://pypi.org/project/qwencleo-asr/)",
    "",
    "> **Runtime → Change runtime type → GPU** before running. Then run the cells "
    "top to bottom.",
)


def banner(title):
    b = json.loads(json.dumps(BANNER))
    b["source"] = [s.replace("{title}", title) for s in b["source"]]
    return b


def install_cell(extra=""):
    # torch is preinstalled on Colab with a compatible CUDA, so a plain install
    # is fine here (unlike the older-driver workstation case).
    pkg = "qwencleo-asr" + (f"[{extra}]" if extra else "")
    return code(
        "# Install QwenCleo-ASR (Colab already has a CUDA-matched torch)",
        f"!pip install -q {pkg}",
    )


SAMPLE_CELL = code(
    "# Grab a sample Egyptian/code-switching clip to transcribe",
    "import urllib.request, os",
    f"URL = '{SAMPLE_URL}'",
    f"FALLBACK = '{FALLBACK_URL}'",
    "path = 'sample.wav'",
    "try:",
    "    urllib.request.urlretrieve(URL, path)",
    "except Exception:",
    "    urllib.request.urlretrieve(FALLBACK, path)",
    "print('saved', path, os.path.getsize(path), 'bytes')",
)


# ======================================================================== #
# 1. Quick start
# ======================================================================== #
def quickstart():
    cells = [
        banner("Quick Start"),
        md("## 1. Install"),
        install_cell(),
        md("## 2. Get a sample audio clip"),
        SAMPLE_CELL,
        md("## 3. Transcribe",
           "",
           "`language` defaults to `\"Arabic\"`. Pass `language=None` to auto-detect."),
        code(
            "from qwencleo_asr import QwenCleoASR",
            "",
            "asr = QwenCleoASR()                 # downloads the model on first use",
            "result = asr.transcribe('sample.wav')",
            "print(result.text)",
            "print('language:', result.language)",
        ),
        md("## 4. Batch + Egyptian normalization"),
        code(
            "# transcribe a list, and optionally normalize the output",
            "out = asr.transcribe(['sample.wav'], normalize=True)",
            "for r in out:",
            "    print(r.text)",
        ),
        md("## 5. CLI (same model, from the shell)"),
        code(
            "!qwencleo transcribe sample.wav",
        ),
    ]
    write("01_quickstart.ipynb", notebook(cells))


# ======================================================================== #
# 2. Chunked transcription of long audio / streaming-style
# ======================================================================== #
def chunked():
    cells = [
        banner("Chunked Long-Audio Transcription"),
        md("Split long or live audio into overlapping windows and transcribe each "
           "— convenient for captioning without a server.",
           "",
           "> ℹ️ This is **chunked transcription**, not true streaming. For "
           "token-by-token streaming see the **vLLM** notebook."),
        md("## 1. Install"),
        install_cell(),
        md("## 2. Sample audio"),
        SAMPLE_CELL,
        md("## 3. Stream a file window by window"),
        code(
            "from qwencleo_asr import QwenCleoASR, stream_file",
            "",
            "asr = QwenCleoASR()",
            "for chunk in stream_file(asr, 'sample.wav', chunk_s=20, overlap_s=2):",
            "    tag = ' (final)' if chunk.is_final else ''",
            "    print(f'[{chunk.start:.0f}-{chunk.end:.0f}s]{tag} {chunk.text}')",
        ),
        md("## 4. Mic-style frame feeding (ChunkedSession)",
           "",
           "Simulates feeding audio frames (e.g. from a microphone) and pulling "
           "transcripts as enough audio accumulates."),
        code(
            "import soundfile as sf",
            "import numpy as np",
            "from qwencleo_asr import QwenCleoASR, ChunkedSession",
            "",
            "asr = QwenCleoASR()",
            "audio, sr = sf.read('sample.wav', dtype='float32')",
            "if audio.ndim > 1:",
            "    audio = audio.mean(axis=1)",
            "",
            "sess = ChunkedSession(asr, sr=sr, chunk_s=8)",
            "# feed in 1-second frames",
            "for i in range(0, len(audio), sr):",
            "    frame = audio[i:i+sr]",
            "    for chunk in sess.add(frame):",
            "        print(chunk.text)",
            "for chunk in sess.flush():",
            "    print(chunk.text)",
        ),
    ]
    write("02_chunked_transcription.ipynb", notebook(cells))


# ======================================================================== #
# 3. FastAPI server
# ======================================================================== #
def server():
    cells = [
        banner("FastAPI Server"),
        md("Run the QwenCleo FastAPI server inside Colab and call it over HTTP."),
        md("## 1. Install (server extras) + get the repo code"),
        code(
            "!pip install -q 'qwencleo-asr[server]'",
            "# the server app lives in the GitHub repo",
            "!git clone -q https://github.com/MohammedAly22/qwencleo-asr.git",
            "%cd qwencleo-asr",
        ),
        md("## 2. Launch the server in the background"),
        code(
            "import subprocess, os, time",
            "env = dict(os.environ, QWENCLEO_MODEL='mohammedaly22/QwenCleo-ASR')",
            "proc = subprocess.Popen(",
            "    ['uvicorn', 'server.app:app', '--host', '0.0.0.0', '--port', '8000'],",
            "    env=env)",
            "print('starting server (first run downloads the model)...')",
            "time.sleep(60)   # give it time to load the model",
        ),
        md("## 3. Health check"),
        code(
            "import requests",
            "print(requests.get('http://localhost:8000/health').json())",
        ),
        md("## 4. Transcribe a file via the API"),
        SAMPLE_CELL,
        code(
            "import requests",
            "with open('sample.wav', 'rb') as f:",
            "    r = requests.post('http://localhost:8000/v1/transcribe',",
            "                      files={'file': f},",
            "                      data={'language': 'Arabic'})",
            "print(r.json())",
        ),
        md("## 5. Stop the server"),
        code("proc.terminate()"),
    ]
    write("03_fastapi_server.ipynb", notebook(cells))


# ======================================================================== #
# 4. Gradio demo
# ======================================================================== #
def gradio():
    cells = [
        banner("Gradio Demo"),
        md("Launch the QwenCleo Gradio app with a public share link — upload audio "
           "or record from the mic, right in the browser."),
        md("## 1. Install (app extras) + clone the repo"),
        code(
            "!pip install -q 'qwencleo-asr[app]'",
            "!git clone -q https://github.com/MohammedAly22/qwencleo-asr.git",
            "%cd qwencleo-asr",
        ),
        md("## 2. Launch with a public link",
           "",
           "We import the app module and call `launch(share=True)` so Colab gives "
           "you a public URL."),
        code(
            "import sys; sys.path.insert(0, 'app')",
            "import gradio_app           # builds the QwenCleoASR + UI",
            "gradio_app.demo.queue().launch(share=True)",
        ),
        md("Click the public `gradio.live` link that appears above. Upload a clip "
           "or use the mic, then **Transcribe**."),
    ]
    write("04_gradio_demo.ipynb", notebook(cells))


# ======================================================================== #
# 5. vLLM serving + true streaming
# ======================================================================== #
def vllm():
    cells = [
        banner("vLLM Serving & True Streaming"),
        md("QwenCleo inherits Qwen3-ASR's **token-by-token streaming** via vLLM.",
           "",
           "> ⚠️ Needs a recent CUDA. On Colab pick a **GPU** runtime; vLLM's "
           "Qwen3-ASR support is on the **nightly** build, which is large — the "
           "install cell can take several minutes."),
        md("## 1. Install vLLM (nightly) + the client"),
        code(
            "# vLLM nightly has day-0 Qwen3-ASR support",
            "!pip install -q -U vllm --pre \\",
            "    --extra-index-url https://wheels.vllm.ai/nightly \\",
            "    --index-strategy unsafe-best-match || pip install -q -U vllm",
            "!pip install -q 'qwencleo-asr[vllm-client]'",
        ),
        md("## 2. Start a vLLM server in the background"),
        code(
            "import subprocess, time, os",
            "proc = subprocess.Popen(",
            "    ['vllm', 'serve', 'mohammedaly22/QwenCleo-ASR'])",
            "print('starting vLLM (downloads + loads the model)...')",
            "# wait for the server to come up",
            "import requests",
            "for _ in range(60):",
            "    try:",
            "        requests.get('http://localhost:8000/v1/models', timeout=2)",
            "        print('server up'); break",
            "    except Exception:",
            "        time.sleep(10)",
        ),
        md("## 3. Sample audio"),
        SAMPLE_CELL,
        md("## 4. TRUE streaming — deltas arrive token by token"),
        code(
            "from qwencleo_asr import stream_vllm",
            "",
            "for delta in stream_vllm('sample.wav', language='Arabic'):",
            "    print(delta, end='', flush=True)",
            "print()",
        ),
        md("## 5. One-shot via the OpenAI-compatible API"),
        code(
            "from qwencleo_asr import transcribe_vllm",
            "print(transcribe_vllm('sample.wav'))",
        ),
        md("## 6. OpenAI SDK directly (the documented form)"),
        code(
            "from openai import OpenAI",
            "client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')",
            "with open('sample.wav','rb') as f:",
            "    print(client.audio.transcriptions.create(",
            "        model='mohammedaly22/QwenCleo-ASR', file=f.read()).text)",
        ),
        md("## 7. Stop the server"),
        code("proc.terminate()"),
    ]
    write("05_vllm_streaming.ipynb", notebook(cells))


if __name__ == "__main__":
    quickstart()
    chunked()
    server()
    gradio()
    vllm()
    print("\nAll notebooks written to", HERE)
