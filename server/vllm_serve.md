# Serving & Streaming QwenCleo-ASR with vLLM

QwenCleo-ASR is a fine-tune of **Qwen3-ASR-1.7B**, so it inherits Qwen3-ASR's
vLLM support — including **true token-by-token streaming**. Everywhere the
upstream docs say `Qwen/Qwen3-ASR-1.7B`, use `mohammedaly22/QwenCleo-ASR`.

- ⚡ **Real streaming** — token-by-token, via vLLM (not chunking).
- 🌐 **OpenAI-compatible** — chat-completions, transcription API, and cURL.
- 🖥️ **Mic web demo** — the `qwen-asr-demo-streaming` Flask app.
- 📦 **Offline** — in-process `vllm.LLM`.

---

## Install vLLM

> ⚠️ **CUDA 13 required.** The vLLM **nightly** that carries Qwen3-ASR support is
> built against **CUDA 13** (`libcudart.so.13`). On a CUDA 12.x box it fails to
> import (`ImportError: libcudart.so.13`). Run vLLM on a CUDA-13 runtime; on
> CUDA 12.x use the [FastAPI server](app.py) instead — it needs no vLLM and gives
> the same transcriptions (just without token-by-token streaming).

vLLM provides day-0 support for the Qwen3-ASR architecture. Use the **nightly**
wheel (recommended via `uv`):

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]"      # audio dependencies
```

Or, with the QwenCleo extra:

```bash
pip install "qwencleo-asr[vllm]"
```

---

## Online serving

```bash
vllm serve mohammedaly22/QwenCleo-ASR
# 2x H100 tensor parallel:
vllm serve mohammedaly22/QwenCleo-ASR --tensor-parallel-size 2 --dtype bfloat16
```

Once up, interact in several ways.

### Python — via QwenCleo's helpers

```python
from qwencleo_asr import stream_vllm, transcribe_vllm

# TRUE streaming — deltas arrive as they are generated
for delta in stream_vllm("clip.wav", language="Arabic"):
    print(delta, end="", flush=True)

# one-shot
print(transcribe_vllm("clip.wav"))
```

…or the same thing straight off the model object:

```python
from qwencleo_asr import QwenCleoASR
asr = QwenCleoASR()                       # local model for transcribe()
for delta in asr.stream("clip.wav"):      # delegates to the vLLM server
    print(delta, end="", flush=True)
```

### OpenAI SDK — chat completions

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="mohammedaly22/QwenCleo-ASR",
    messages=[{
        "role": "user",
        "content": [{
            "type": "audio_url",
            "audio_url": {"url": "https://.../clip.wav"},
        }],
    }],
)
print(response.choices[0].message.content)
```

### OpenAI SDK — transcription API

```python
import httpx
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
audio = httpx.get("https://.../clip.wav").content
print(client.audio.transcriptions.create(
    model="mohammedaly22/QwenCleo-ASR", file=audio).text)
```

### cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "https://.../clip.wav"}}
      ]}
    ]
  }'
```

---

## Streaming mic demo (web UI)

The upstream `qwen-asr` package ships a Flask mic demo that captures browser
audio, resamples to 16 kHz, and pushes PCM chunks to the model for live
transcription. Point it at QwenCleo:

```bash
qwen-asr-demo-streaming \
  --asr-model-path mohammedaly22/QwenCleo-ASR \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

Then open `http://<your-ip>:8000` (or via VS Code port forwarding).

---

## Offline inference (no server)

```python
from qwencleo_asr import VLLMOffline

off = VLLMOffline()                       # loads mohammedaly22/QwenCleo-ASR
print(off.transcribe("clip.wav", language="Arabic"))
```

Equivalent to the raw vLLM API:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="mohammedaly22/QwenCleo-ASR")
conversation = [{
    "role": "user",
    "content": [{"type": "audio_url", "audio_url": {"url": "clip.wav"}}],
}]
out = llm.chat(conversation, SamplingParams(temperature=0.01, max_tokens=256))
print(out[0].outputs[0].text)
```

---

## Notes

- **Streaming** (`stream_vllm` / `asr.stream`) does **not** support batching or
  timestamps — same limitation as upstream Qwen3-ASR streaming.
- Egyptian normalization is not applied server-side; do it client-side with
  `from qwencleo_asr import normalize`.
- vLLM nightly tracks fast-moving wheels; pin a known-good build for production.
