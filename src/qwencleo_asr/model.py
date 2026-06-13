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
        if quiet:
            _quiet_transformers()

        import torch
        from qwen_asr import Qwen3ASRModel

        torch_dtype = getattr(torch, dtype)
        self.model_id = model_id
        self.default_language = default_language

        logger.info("loading %s on %s (%s)", model_id, device, dtype)
        self._model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map=device,
            max_new_tokens=max_new_tokens,
        )

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

        single = isinstance(audio, str)
        items = [audio] if single else list(audio)

        results: List[TranscriptionResult] = []
        for path in items:
            out = self._model.transcribe(audio=path, language=lang_arg)
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
