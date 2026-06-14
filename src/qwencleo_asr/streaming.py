"""Chunked transcription for long audio.

NOTE — this is NOT true low-latency streaming. Qwen3-ASR (the base model) *does*
support genuine token-by-token streaming, but only through the **vLLM backend**
(see `server/vllm_serve.md` and the Qwen3-ASR streaming example). What this
module provides instead is *chunked transcription*: long or live audio is split
into overlapping windows and each window is transcribed with the regular
`transcribe()` call, yielding partial text window by window. This is convenient
for captioning long files without a vLLM server, but latency is per-window, not
per-token.

For real streaming, run the vLLM backend.

Two entry points:
  * stream_file(asr, path)  -> generator of per-window transcripts for a long file
  * ChunkedSession(asr)     -> feed raw audio frames (e.g. from a mic) and pull
                               transcripts as enough audio accumulates
"""
from __future__ import annotations

import tempfile
import wave
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

from .model import QwenCleoASR


@dataclass
class StreamChunk:
    index: int
    start: float          # seconds
    end: float            # seconds
    text: str
    is_final: bool = False


def _write_wav(samples: np.ndarray, sr: int) -> str:
    """Write float32/-1..1 or int16 mono samples to a temp wav, return path."""
    if samples.dtype != np.int16:
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).astype(np.int16)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(f.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return f.name


def stream_file(
    asr: QwenCleoASR,
    path: str,
    *,
    chunk_s: float = 20.0,
    overlap_s: float = 2.0,
    sr: int = 16000,
    language: Optional[str] = "__default__",
) -> Iterator[StreamChunk]:
    """Yield StreamChunk objects as a long file is transcribed window by window.

    Overlap reduces word loss at boundaries. The naive concatenation of chunk
    texts is a good transcript; for best quality dedup overlapping words
    downstream if needed.
    """
    import soundfile as sf

    audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        # lightweight resample via torchaudio if rates differ
        import torch
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio), file_sr, sr).numpy()

    win = int(chunk_s * sr)
    hop = int((chunk_s - overlap_s) * sr)
    total = len(audio)
    idx = 0
    pos = 0
    while pos < total:
        seg = audio[pos: pos + win]
        if len(seg) < int(0.3 * sr):  # skip <0.3s tail
            break
        tmp = _write_wav(seg, sr)
        r = asr.transcribe(tmp, language=language)
        is_final = pos + win >= total
        yield StreamChunk(
            index=idx,
            start=pos / sr,
            end=min(pos + win, total) / sr,
            text=r.text,
            is_final=is_final,
        )
        idx += 1
        pos += hop


@dataclass
class ChunkedSession:
    """Accumulate raw audio frames (e.g. from a microphone) and transcribe once
    enough audio has buffered. Near-real-time captioning by *chunking* — latency
    is per-window, not per-token. For true streaming use the vLLM backend.

    Usage
    -----
    >>> sess = ChunkedSession(asr, sr=16000)
    >>> for frame in mic_frames():        # np.ndarray float32 mono @ sr
    ...     for chunk in sess.add(frame):
    ...         print(chunk.text)
    >>> for chunk in sess.flush():         # transcribe the remainder
    ...     print(chunk.text)
    """
    asr: QwenCleoASR
    sr: int = 16000
    chunk_s: float = 8.0
    language: Optional[str] = "__default__"
    _buf: List[np.ndarray] = field(default_factory=list)
    _buffered: int = 0
    _idx: int = 0
    _t: float = 0.0

    def add(self, frame: np.ndarray) -> Iterator[StreamChunk]:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        self._buf.append(frame)
        self._buffered += len(frame)
        if self._buffered >= int(self.chunk_s * self.sr):
            yield from self._emit(is_final=False)

    def flush(self) -> Iterator[StreamChunk]:
        if self._buffered > int(0.3 * self.sr):
            yield from self._emit(is_final=True)

    def _emit(self, *, is_final: bool) -> Iterator[StreamChunk]:
        samples = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
        tmp = _write_wav(samples, self.sr)
        r = self.asr.transcribe(tmp, language=self.language)
        dur = len(samples) / self.sr
        chunk = StreamChunk(
            index=self._idx, start=self._t, end=self._t + dur,
            text=r.text, is_final=is_final,
        )
        self._idx += 1
        self._t += dur
        self._buf.clear()
        self._buffered = 0
        yield chunk


# Backwards-compatible alias. ChunkedSession is the accurate name; the old name
# is kept so existing imports keep working.
StreamingSession = ChunkedSession
