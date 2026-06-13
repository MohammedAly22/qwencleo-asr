"""QwenCleo-ASR FastAPI server.

Run (after `pip install qwencleo-asr[server]` or installing requirements):

    QWENCLEO_MODEL=mohammedaly22/QwenCleo-ASR \
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health
    POST /v1/transcribe            (multipart file upload)
    POST /v1/audio/transcriptions  (OpenAI-compatible-ish: multipart `file`)
"""
from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from qwencleo_asr import QwenCleoASR

MODEL_ID = os.environ.get("QWENCLEO_MODEL", "mohammedaly22/QwenCleo-ASR")
DEVICE = os.environ.get("QWENCLEO_DEVICE", "cuda:0")
DTYPE = os.environ.get("QWENCLEO_DTYPE", "bfloat16")

app = FastAPI(title="QwenCleo-ASR", version="0.1.0")
_asr: QwenCleoASR | None = None


def get_asr() -> QwenCleoASR:
    global _asr
    if _asr is None:
        _asr = QwenCleoASR(MODEL_ID, device=DEVICE, dtype=DTYPE)
    return _asr


@app.on_event("startup")
def _warm():
    get_asr()  # load weights at boot, not on first request


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


async def _save_upload(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(await file.read())
    tmp.flush()
    return tmp.name


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("Arabic"),
    normalize: bool = Form(False),
):
    path = await _save_upload(file)
    asr = get_asr()
    lang = None if language in ("None", "", "auto") else language
    r = asr.transcribe(path, language=lang, normalize=normalize)
    return {"text": r.text, "language": r.language}


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    language: str = Form("Arabic"),
):
    """Loosely OpenAI-compatible shape: returns {"text": ...}."""
    path = await _save_upload(file)
    asr = get_asr()
    lang = None if language in ("None", "", "auto") else language
    r = asr.transcribe(path, language=lang)
    return JSONResponse({"text": r.text})
