"""QwenCleo-ASR — Egyptian Arabic & code-switching speech recognition.

Built on Qwen3-ASR-1.7B, fine-tuned for Egyptian dialect and Arabic/English
code-switching.

    from qwencleo_asr import QwenCleoASR
    asr = QwenCleoASR()
    print(asr.transcribe("audio.wav").text)
"""
from .model import QwenCleoASR, TranscriptionResult, DEFAULT_MODEL_ID
from .streaming import stream_file, StreamingSession, StreamChunk
from .normalize import normalize

__version__ = "0.1.0"

__all__ = [
    "QwenCleoASR",
    "TranscriptionResult",
    "DEFAULT_MODEL_ID",
    "stream_file",
    "StreamingSession",
    "StreamChunk",
    "normalize",
    "__version__",
]
