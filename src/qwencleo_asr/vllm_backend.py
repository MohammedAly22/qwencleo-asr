"""Real streaming + vLLM helpers for QwenCleo-ASR.

QwenCleo (like its base Qwen3-ASR) supports genuine token-by-token streaming
through **vLLM**. This module provides:

  * stream_vllm(...)        -> generator yielding text deltas from a running
                              vLLM OpenAI-compatible server (true streaming)
  * transcribe_vllm(...)    -> one-shot transcription via the same server
  * VLLMOffline             -> in-process offline inference with vllm.LLM

Start a server first (see server/vllm_serve.md):

    vllm serve mohammedaly22/QwenCleo-ASR

Then point these helpers at http://localhost:8000/v1.
"""
from __future__ import annotations

import base64
from typing import Iterator, Optional

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "mohammedaly22/QwenCleo-ASR"


def _audio_to_data_url(audio: str) -> str:
    """Turn a local path or URL into something usable as audio_url.url.

    Remote URLs are passed through. Local files are inlined as a base64 data URI
    so the server doesn't need filesystem access to the client's machine.
    """
    if audio.startswith(("http://", "https://")):
        return audio
    with open(audio, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _client(base_url: str, api_key: str = "EMPTY"):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "vLLM streaming needs the openai client: pip install openai"
        ) from e
    return OpenAI(base_url=base_url, api_key=api_key)


def stream_vllm(
    audio: str,
    *,
    base_url: Optional[str] = None,
    port: int = 8000,
    model: str = DEFAULT_MODEL,
    language: Optional[str] = "Arabic",
    api_key: str = "EMPTY",
    max_tokens: int = 256,
) -> Iterator[str]:
    """Yield text deltas as the vLLM server transcribes — TRUE streaming.

    base_url: full OpenAI base URL; if None, built from `port`
              (http://localhost:<port>/v1).

    The model emits a `language X<asr_text>` prefix before the transcript; we
    strip it on the fly so callers get only the spoken text.

    Example
    -------
    >>> for delta in stream_vllm("clip.wav", port=8000):
    ...     print(delta, end="", flush=True)
    """
    if base_url is None:
        base_url = f"http://localhost:{port}/v1"
    client = _client(base_url, api_key)
    content = [{"type": "audio_url",
                "audio_url": {"url": _audio_to_data_url(audio)}}]
    if language and language != "None":
        # nudge the decoder toward the target language
        content.insert(0, {"type": "text", "text": f"language {language}"})

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        stream=True,
    )

    # Strip the leading "language X<asr_text>" prefix, which can span several
    # streamed chunks. Buffer until we pass the marker, then yield cleanly.
    marker = "<asr_text>"
    buf = ""
    passed = False
    for event in stream:
        delta = event.choices[0].delta.content
        if not delta:
            continue
        if passed:
            yield delta
            continue
        buf += delta
        idx = buf.find(marker)
        if idx != -1:
            passed = True
            tail = buf[idx + len(marker):]
            if tail:
                yield tail
        # else: still inside the prefix; keep buffering (don't yield)
    if not passed and buf:
        # no marker ever seen — emit whatever we buffered (be safe)
        yield buf


def transcribe_vllm(
    audio: str,
    *,
    base_url: Optional[str] = None,
    port: int = 8000,
    model: str = DEFAULT_MODEL,
    api_key: str = "EMPTY",
) -> str:
    """One-shot transcription via the vLLM OpenAI transcription API."""
    import httpx
    if base_url is None:
        base_url = f"http://localhost:{port}/v1"
    client = _client(base_url, api_key)
    if audio.startswith(("http://", "https://")):
        data = httpx.get(audio).content
    else:
        with open(audio, "rb") as f:
            data = f.read()
    tr = client.audio.transcriptions.create(model=model, file=data)
    return tr.text


class VLLMOffline:
    """In-process offline inference with vllm.LLM (no server).

    >>> off = VLLMOffline()
    >>> print(off.transcribe("clip.wav"))
    """

    def __init__(self, model: str = DEFAULT_MODEL, **llm_kwargs):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError("pip install vllm to use VLLMOffline") from e
        self._SamplingParams = SamplingParams
        self.llm = LLM(model=model, **llm_kwargs)

    def transcribe(self, audio: str, *, language: Optional[str] = "Arabic",
                   max_tokens: int = 256, temperature: float = 0.01) -> str:
        content = [{"type": "audio_url",
                    "audio_url": {"url": _audio_to_data_url(audio)}}]
        if language and language != "None":
            content.insert(0, {"type": "text", "text": f"language {language}"})
        conversation = [{"role": "user", "content": content}]
        sp = self._SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.chat(conversation, sampling_params=sp)
        return outputs[0].outputs[0].text
