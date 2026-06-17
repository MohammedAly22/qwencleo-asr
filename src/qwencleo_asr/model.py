"""QwenCleo-ASR core model wrapper.

Thin layer over Qwen3ASRModel (the `qwen-asr` package) that:
  * defaults to the QwenCleo-ASR checkpoint on the Hub,
  * defaults the language hint to Arabic (the model is tuned for Egyptian
    Arabic + code-switching),
  * returns a small, stable result object,
  * optionally post-normalizes the hypothesis.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from .normalize import normalize as _normalize


DEFAULT_MODEL_ID = "mohammedaly22/QwenCleo-ASR"

logger = logging.getLogger("qwencleo_asr")


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    audio: Optional[str] = None

    def __str__(self) -> str:  # convenient printing
        return self.text


def _quiet_transformers():
    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    warnings.filterwarnings("ignore", category=UserWarning)


class QwenCleoASR:
    """High-level Egyptian / code-switching ASR.

    Example
    -------
    >>> from qwencleo_asr import QwenCleoASR
    >>> asr = QwenCleoASR()                 # loads mohammedaly22/QwenCleo-ASR
    >>> print(asr.transcribe("clip.wav").text)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        default_language: Optional[str] = "Arabic",
        quiet: bool = True,
    ):
        # Store config; defer the heavy local-model load until transcribe() is
        # actually called. This lets `stream()` (which only talks to the vLLM
        # server over HTTP) work without importing/loading qwen_asr at all.
        self.model_id = model_id
        self.default_language = default_language
        self._device = device
        self._dtype = dtype
        self._max_new_tokens = max_new_tokens
        self._quiet = quiet
        self._model = None

    def _ensure_model(self):
        """Lazily load the local HF model on first transcribe() call."""
        if self._model is not None:
            return self._model
        if self._quiet:
            _quiet_transformers()
        import torch
        from qwen_asr import Qwen3ASRModel

        logger.info("loading %s on %s (%s)", self.model_id, self._device, self._dtype)
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            dtype=getattr(torch, self._dtype),
            device_map=self._device,
            max_new_tokens=self._max_new_tokens,
        )
        return self._model

    # ------------------------------------------------------------------ #
    def transcribe(
        self,
        audio: Union[str, Sequence[str]],
        *,
        language: Optional[str] = "__default__",
        normalize: bool = False,
    ) -> Union[TranscriptionResult, List[TranscriptionResult]]:
        """Transcribe one path/URL or a list of them.

        language:
            "Arabic" (default), any Qwen3-ASR language name, or None to
            auto-detect. Pass nothing to use the instance default.
        normalize:
            if True, apply Egyptian-aware normalization to the hypothesis.
        """
        if language == "__default__":
            language = self.default_language
        lang_arg = None if language in (None, "None") else language

        model = self._ensure_model()
        single = isinstance(audio, str)
        items = [audio] if single else list(audio)

        results: List[TranscriptionResult] = []
        for path in items:
            out = model.transcribe(audio=path, language=lang_arg)
            r = out[0]
            text = r.text
            if normalize:
                text = _normalize(text)
            results.append(
                TranscriptionResult(
                    text=text,
                    language=getattr(r, "language", None),
                    audio=path,
                )
            )
        return results[0] if single else results

    # convenience alias
    __call__ = transcribe

    # ------------------------------------------------------------------ #
    def stream(
        self,
        audio: str,
        port: int = 8000,
        *,
        base_url: Optional[str] = None,
        language: Optional[str] = "__default__",
        model_id: Optional[str] = None,
        max_tokens: int = 256,
    ):
        """TRUE token-by-token streaming via a running vLLM server.

        Start the server first (see server/vllm_serve.md):

            vllm serve mohammedaly22/QwenCleo-ASR

        Then:

            asr = QwenCleoASR()
            for delta in asr.stream("clip.wav", port=8000):
                print(delta, end="", flush=True)

        audio    : path or URL to the audio.
        port     : port the vLLM server listens on (default 8000). Ignored if
                   `base_url` is given explicitly.
        Unlike `transcribe()` (which runs the local HF model) this connects to
        the vLLM OpenAI-compatible endpoint and yields the transcript text as it
        is generated (the `language X<asr_text>` prefix is stripped for you).
        """
        from .vllm_backend import stream_vllm

        if language == "__default__":
            language = self.default_language
        return stream_vllm(
            audio,
            base_url=base_url,
            port=port,
            model=model_id or self.model_id,
            language=language,
            max_tokens=max_tokens,
        )
