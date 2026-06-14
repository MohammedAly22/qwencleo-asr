"""QwenCleo-ASR — Egyptian Arabic & code-switching speech recognition.

Built on Qwen3-ASR-1.7B, fine-tuned for Egyptian dialect and Arabic/English
code-switching.

    from qwencleo_asr import QwenCleoASR
    asr = QwenCleoASR()
    print(asr.transcribe("audio.wav").text)
"""
from .model import QwenCleoASR, TranscriptionResult, DEFAULT_MODEL_ID
from .streaming import stream_file, ChunkedSession, StreamingSession, StreamChunk
from .normalize import normalize

# vLLM-backed helpers are imported lazily (they need optional deps). Importing
# the names here is cheap because vllm_backend defers its heavy imports.
from .vllm_backend import stream_vllm, transcribe_vllm, VLLMOffline

__version__ = "0.1.0"

__all__ = [
    "QwenCleoASR",
    "TranscriptionResult",
    "DEFAULT_MODEL_ID",
    "stream_file",
    "ChunkedSession",
    "StreamingSession",  # backwards-compatible alias of ChunkedSession
    "StreamChunk",
    "normalize",
    # vLLM / true streaming
    "stream_vllm",
    "transcribe_vllm",
    "VLLMOffline",
    "__version__",
]
