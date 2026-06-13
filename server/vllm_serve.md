# Serving QwenCleo-ASR with vLLM

QwenCleo-ASR is a fine-tune of **Qwen3-ASR-1.7B**, an audio encoder + Qwen3
decoder. vLLM serving of Qwen3-ASR follows the same path as upstream Qwen3-ASR —
QwenCleo only swaps in the fine-tuned weights, so use the model id
`mohammedaly22/QwenCleo-ASR` wherever the upstream docs say `Qwen/Qwen3-ASR-1.7B`.

> ⚠️ vLLM support for Qwen3-ASR depends on your vLLM version shipping the
> `qwen3_asr` (audio) architecture. Check `vllm --version` and the
> [vLLM supported-models list](https://docs.vllm.ai/en/latest/models/supported_models.html)
> before relying on this in production. If your build lacks it, use the FastAPI
> server (`server/app.py`) which always works.

## Install

```bash
pip install "qwencleo-asr[vllm]"
# or, in the cloned repo:
pip install -r requirements-serving.txt vllm
```

## Launch an OpenAI-compatible server

```bash
vllm serve mohammedaly22/QwenCleo-ASR \
  --task transcription \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

For 2× H100 tensor parallelism:

```bash
vllm serve mohammedaly22/QwenCleo-ASR \
  --task transcription --tensor-parallel-size 2 \
  --dtype bfloat16 --port 8000
```

## Call it (OpenAI audio API shape)

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@clip.wav \
  -F model=mohammedaly22/QwenCleo-ASR \
  -F language=ar
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
with open("clip.wav", "rb") as f:
    print(client.audio.transcriptions.create(
        model="mohammedaly22/QwenCleo-ASR", file=f, language="ar").text)
```

## Notes

- Egyptian normalization is **not** applied server-side by vLLM; do it client
  side with `from qwencleo_asr import normalize`.
- Throughput scales with batch size; vLLM batches concurrent requests
  automatically. For offline bulk transcription the FastAPI server looping over
  files is simpler and nearly as fast on a single GPU.
