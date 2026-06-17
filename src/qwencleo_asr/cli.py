"""QwenCleo-ASR CLI.

Commands:
  qwencleo transcribe <audio...>     local HF model, batch transcription
  qwencleo stream <audio>            chunked long-audio transcription (no server)
  qwencleo install-vllm              install the vLLM nightly (cu129) for serving
  qwencleo serve                     launch the vLLM OpenAI server
  qwencleo stream-vllm <audio>       TRUE token-by-token streaming via the server
"""
from __future__ import annotations

import argparse
import os
import subprocess
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


# vLLM nightly install (cu129) — the only build with Qwen3-ASR support.
VLLM_INSTALL = [
    [sys.executable, "-m", "pip", "install", "-U", "uv"],
    ["uv", "pip", "install", "--system", "-U", "vllm", "--pre",
     "--extra-index-url", "https://wheels.vllm.ai/nightly/cu129",
     "--extra-index-url", "https://download.pytorch.org/whl/cu129",
     "--index-strategy", "unsafe-best-match"],
    ["uv", "pip", "install", "--system", "vllm[audio]", "openai", "httpx"],
]


def main(argv=None):
    ap = argparse.ArgumentParser("qwencleo", description="QwenCleo-ASR CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="Transcribe audio with the local model.")
    _add_common(t)
    t.add_argument("audio", nargs="+", help="Audio file path(s) or URL(s).")

    s = sub.add_parser("stream", help="Chunked transcription of a long file "
                       "(windowed; not true streaming — use stream-vllm for that).")
    _add_common(s)
    s.add_argument("audio", help="Audio file path.")
    s.add_argument("--chunk-s", type=float, default=20.0)
    s.add_argument("--overlap-s", type=float, default=2.0)

    iv = sub.add_parser("install-vllm",
                        help="Install the vLLM nightly (cu129) needed for serving.")
    iv.add_argument("--dry-run", action="store_true", help="Print commands only.")

    sv = sub.add_parser("serve", help="Launch the vLLM OpenAI-compatible server.")
    sv.add_argument("--model", default=DEFAULT_MODEL_ID)
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    sv.add_argument("--max-model-len", type=int, default=4096)
    sv.add_argument("--dtype", default="bfloat16")

    sx = sub.add_parser("stream-vllm",
                        help="TRUE token-by-token streaming via a running server.")
    sx.add_argument("audio", help="Audio file path or URL.")
    sx.add_argument("--model", default=DEFAULT_MODEL_ID)
    sx.add_argument("--port", type=int, default=8000)
    sx.add_argument("--language", default="Arabic", help="'None' to auto-detect.")
    sx.add_argument("--max-tokens", type=int, default=256)

    args = ap.parse_args(argv)

    # ---- commands that DON'T need the local model -------------------------
    if args.cmd == "install-vllm":
        for cmd in VLLM_INSTALL:
            print("+", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)
        if not args.dry_run:
            print("\n✅ vLLM nightly installed. Serve with:  qwencleo serve")
        return 0

    if args.cmd == "serve":
        # FlashInfer's sampler JIT-needs nvcc which may be absent; disable it.
        env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0")
        cmd = ["vllm", "serve", args.model,
               "--port", str(args.port),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-model-len", str(args.max_model_len),
               "--dtype", args.dtype]
        print("+", " ".join(cmd))
        return subprocess.run(cmd, env=env).returncode

    if args.cmd == "stream-vllm":
        from .vllm_backend import stream_vllm
        lang = None if args.language == "None" else args.language
        for delta in stream_vllm(args.audio, port=args.port, model=args.model,
                                 language=lang, max_tokens=args.max_tokens):
            print(delta, end="", flush=True)
        print()
        return 0

    # ---- commands that DO need the local model ----------------------------
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
