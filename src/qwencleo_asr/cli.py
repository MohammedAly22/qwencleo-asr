"""Command-line interface: `qwencleo transcribe ...` and `qwencleo stream ...`."""
from __future__ import annotations

import argparse
import sys

from .model import QwenCleoASR, DEFAULT_MODEL_ID


def _add_common(p):
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF id or local path.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--language", default="Arabic",
                   help="Language hint. Use 'None' for auto-detect.")
    p.add_argument("--normalize", action="store_true",
                   help="Apply Egyptian-aware normalization to output.")
    p.add_argument("--max-new-tokens", type=int, default=256)


def main(argv=None):
    ap = argparse.ArgumentParser("qwencleo", description="QwenCleo-ASR CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="Transcribe one or more audio files.")
    _add_common(t)
    t.add_argument("audio", nargs="+", help="Audio file path(s) or URL(s).")

    s = sub.add_parser("stream", help="Chunked transcription of a long file "
                       "(windowed; not true streaming — use vLLM for that).")
    _add_common(s)
    s.add_argument("audio", help="Audio file path.")
    s.add_argument("--chunk-s", type=float, default=20.0)
    s.add_argument("--overlap-s", type=float, default=2.0)

    args = ap.parse_args(argv)

    asr = QwenCleoASR(
        args.model, device=args.device, dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        default_language=None if args.language == "None" else args.language,
    )

    if args.cmd == "transcribe":
        results = asr.transcribe(args.audio, normalize=args.normalize)
        if not isinstance(results, list):
            results = [results]
        for r in results:
            if len(args.audio) > 1:
                print(f"# {r.audio}")
            print(r.text)
        return 0

    if args.cmd == "stream":
        from .streaming import stream_file
        full = []
        for chunk in stream_file(asr, args.audio,
                                 chunk_s=args.chunk_s, overlap_s=args.overlap_s,
                                 language=asr.default_language):
            tag = " (final)" if chunk.is_final else ""
            print(f"[{chunk.start:6.1f}-{chunk.end:6.1f}s]{tag} {chunk.text}",
                  flush=True)
            full.append(chunk.text)
        print("\n--- full ---")
        print(" ".join(full))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
