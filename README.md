<div align="center">

# 🎙️ QwenCleo-ASR

### The best open-source model for Egyptian Arabic & code-switching speech recognition

*Built on [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), fine-tuned for Egyptian dialect and Arabic ↔ English code-switching.*

[![🤗 Model](https://img.shields.io/badge/🤗%20Model-mohammedaly22/QwenCleo--ASR-D4AF37)](https://huggingface.co/mohammedaly22/QwenCleo-ASR)
[![PyPI](https://img.shields.io/badge/PyPI-qwencleo--asr-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/qwencleo-asr/)
[![GitHub](https://img.shields.io/badge/GitHub-qwencleo--asr-181717?logo=github&logoColor=white)](https://github.com/MohammedAly22/qwencleo-asr)
[![Base model](https://img.shields.io/badge/Base-Qwen3--ASR--1.7B-615CED)](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/qwencleo-asr/blob/main/examples/01_quickstart.ipynb)

</div>


<img src="assets/QwenCleo-ASR-Banner.png" alt="QwenCleo-ASR" width="100%"/>


---

> **QwenCleo** — the name carries three meanings: **Qwen**, the powerful base model it is built on; **Queen**, signalling a model that reigns over its domain; and **Cleo**, for **Cleopatra**, the queen of Egypt — because this model is tailored for **Egyptian** Arabic. 👑🏺

**QwenCleo-ASR is, to our knowledge, the best open-source ASR model for Egyptian Arabic and Arabic/English code-switching.** It cuts the error rate of the strong Qwen3-ASR base **roughly in half**, and correctly keeps embedded English tech/loan words in Latin script (`engineering`, `download`, `React`, `at least`) instead of mangling them into broken Arabic.

- 🎯 **Egyptian dialect** — tuned on hundreds of hours of Egyptian podcast speech.
- 🔀 **Code-switching** — keeps English terms in `code`-script, Arabic in Arabic.
- 🥇 **State-of-the-art (open)** — beats Qwen3-ASR base, NVIDIA Nemotron, Cohere, and every Whisper variant on our Egyptian + CS test set.
- 📦 **`pip install qwencleo-asr`** — inference & chunked long-audio transcription in three lines.
- ⚡ **Real streaming** — token-by-token via vLLM (`asr.stream(...)`), plus a mic web demo.
- 🚀 **Serving** — FastAPI server, Gradio demo, OpenAI-compatible vLLM endpoint.

---

## 📊 Results

WER / CER (%) on an Egyptian-Arabic + code-switching test set (3,699 utterances).
**Lower is better.** All models scored with the same Egyptian-aware normalization.

![Benchmark overview](assets/QwenCleo-ASR-Benchmark.png)

<div align="center">

| Rank | Model | Params | WER all | CER all | WER · AR | CER · AR | WER · CS | CER · CS |
|:---:|---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 | **QwenCleo-ASR** | 1.7B | **19.85** | **10.52** | **19.08** | **10.30** | **20.75** | **10.81** |
| 🥈 | Arabic-Whisper-CodeSwitching | 1.55B | 37.98 | 17.86 | 38.48 | 18.71 | 36.92 | 16.22 |
| 🥉 | NVIDIA Nemotron-3.5 | 0.6B | 38.04 | 19.72 | 36.23 | 16.43 | 41.44 | 25.66 |
| 4 | Qwen3-ASR-1.7B (base) | 1.7B | 41.51 | 20.86 | 40.59 | 18.52 | 43.20 | 25.04 |
| 5 | Whisper Large-v3 Turbo (FT) | 0.81B | 50.83 | 22.86 | 48.37 | 18.42 | 55.08 | 37.84 |
| 6 | Cohere Transcribe 03-2026 | 2.0B | 53.20 | 39.12 | 47.95 | 33.45 | 63.76 | 49.66 |
| 7 | MasriSwitch-Gemma3n | 8B | 57.30 | 30.19 | 60.38 | 32.91 | 51.32 | 25.13 |
| 8 | Whisper Large-v3 | 1.54B | 63.94 | 39.76 | 49.25 | 22.76 | 59.32 | 31.52 |
| 9 | Whisper Large-v2 | 1.54B | 72.34 | 48.73 | 60.75 | 33.21 | 66.85 | 40.75 |
| 10 | Whisper Large-v3 Turbo | 0.81B | 73.83 | 46.86 | 59.37 | 29.42 | 66.08 | 37.84 |
| 11 | Whisper Medium | 0.76B | 80.46 | 53.19 | 74.77 | 41.76 | 74.15 | 44.90 |
| 12 | Whisper Small | 0.24B | 89.99 | 60.34 | 77.42 | 42.53 | 87.09 | 55.22 |
| 13 | Whisper Tiny | 0.04B | 124.68 | 89.42 | 116.02 | 77.74 | 110.67 | 74.57 |

</div>


## 🗣️ Sample outputs

Real transcriptions from the test set, scored after Egyptian-aware
normalization. Examples span **short, medium, and long** clips in both
**Egyptian Arabic** and **code-switching**. These are picked to be honest — on
some clips other models also do well; QwenCleo's edge is consistency, dialect
fidelity, and keeping English terms in Latin script.

> ✅ = matches ground truth · ⚠️ = minor slips · ❌ = clear errors. **Bold** marks the wrong spans.

### 🔀 Code-switching

**Short**

> **✅ Ground truth** — بحس `style`ه حلو وشكله حلو `charisma` وهو بيلعب

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | بحس `style`ه حلو وشكله حلو كاريزما وهو بيلعب ⚠️ *(charisma → كاريزما)* |
| Qwen3-ASR (base) | أه بحس **طايله** حلو وشكله حلو و**كرزمه** وهو يلعب ❌ |
| Cohere | **اااااااا** بحس ستايله حلو وشكله حلو وكاريزما وهو بيلعب ⚠️ |
| Nemotron | آه بحس ستايله حلو وشكله حلو كاريزما هو بيلعب ⚠️ |
| Arabic-Whisper-CS | بحس `style` و حلو وشكل وحلو و `charisma` وهو بيلعب ⚠️ |
| MasriSwitch | بحس**ستايله** حلو و شكله حلو و كاريزما وهو **بيعلق** ❌ |

**Medium**

> **✅ Ground truth** — ده حصل ازاي قرار اصلا انك تعمل قناة `YouTube`

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | ده حصل ازاي قرار اصلا انك تعمل قناة `YouTube` ✅ |
| Qwen3-ASR (base) | ده حصل إزاي قرار أصلا إنك تعمل قناة **يوتيوب** ⚠️ |
| Cohere | ده حصل ازاي قرار اصلا انك تعمل قناه **يوتيوب** ⚠️ |
| Nemotron | ده حصل إزاي قرار أصلاً إنك تعمل قناة **يوتيوب** ⚠️ |
| Arabic-Whisper-CS | **دا** حصل ازاي قرار اصلا انك تعمل قناة **يوتيوب** ⚠️ |
| MasriSwitch | ده حصل ازاي قرار أصلا إنك تعمل قناة `YouTube` ✅ |

> **✅ Ground truth** — خليني اوضح بس هدفها ان الطالب يجي هو `already`

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | خليني اوضح بس هدفها ان الطالب يجي هو `already` ✅ |
| Qwen3-ASR (base) | **خلينا** أوضح بس **هدفه** إن الطالب يجي هو **أرريدي** ❌ |
| Cohere | خليني اوضح بس هدفها ان الطالب يجي هو **اوردي** ❌ |
| Nemotron | **خلينا وضح** بس هدفها إن الطالب يجي هو **أوردي** ❌ |
| Arabic-Whisper-CS | خليني اوضح بس هدفها ان الطالب **ييجي** هو `already` ✅ |
| MasriSwitch | خليني أوضح بس **الأهداف** إن الطالب **بيجي** هو `already` ⚠️ |

**Long**

> **✅ Ground truth** — ما هو الفكرة جاية فين النهارده `average` العربية اللي ممكن تجيبها من أوروبا ما بين 23 ل 28000 `dollars`

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | ما هو الفكره جايه فين النهارده `average` العربيه اللي ممكن تجيبها من اوروبا ما بين 23 ل 28000 `dollars` ✅ |
| Qwen3-ASR (base) | ما هو الفكرة جاية فين النهاردة **أفرج** العربية اللي ممكن **تجابها** من أوروبا ما بين 23 ل 28000 دولار ⚠️ |
| Cohere | ما هو الفكره جايه فين النهارده **افريقيا** العربيه اللي ممكن تجيبها من اوروبا ما بين 23 ل 28000 دولار ❌ |
| Nemotron | ما هو الفكرة جاية فين النهار ده **أفريد** العربية اللي ممكن تجيبها من أوروبا ما بين 23 ل 28000 دولار **ثلاثة** ❌ |
| Arabic-Whisper-CS | ما هو الفكرة جاية فين النهار **دا** `average` العربية اللي ممكن تجيبها من أوروبا ما بين 23 ل 28 ألف دولار **تقريبا** ⚠️ |
| MasriSwitch | ما هو الفكرة جاية فين النهاردة `average` العربية اللي ممكن تجيبها من أوروبا ما بين 23 لـ28 ألف دولار **تقريبًا** ⚠️ |

> **✅ Ground truth** — فبتديك كل ال`soft skills` بشكل `indirect` حرفيا بتديهالك بالمعلقه ليه بتتعلم تشتغل على كل الحاجات بتاعه `Microsoft Excel PowerPoint Word whatever` بعدين كمان بتتعلم ان انت تتكلم مع الناس `public speaking`

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | فبتديك كل ال`soft skills` بشكل `indirect` حرفيا بتديهالك بالمعلقه ليه بتتعلم تشتغل على كل الحاجات بتاعه `Microsoft Excel, PowerPoint, Word` و `whatever` بعدين كمان بتتعلم ان انت تتكلم مع الناس `public speaking` ✅ |
| Arabic-Whisper-CS | فبتديك كل ال `soft skills` بشكل `indirect` حرفيا بتديهالك بالمعلقة ليه بتتعلم تشتغل على كل الحاجات بتاعة `Microsoft, Excel, PowerPoint, Word whatever` بعدين كمان بتتعلم أن أنت ✅ |
| MasriSwitch | فبتديك كل ال `soft skills` بشكل `indirect` حرفيا بتديهالك **بالمعلاة** ليه بتتعلم تشتغل على كل الحاجات بتاعة `Microsoft Excel, PowerPoint, Word whatever` بعدين كمان بتتعلم أن أنت ⚠️ |

### 🇪🇬 Egyptian Arabic

Pure-Arabic clips (no English). After normalization these reflect genuine
word-level accuracy, not spelling/number-format differences.

**Short**

> **✅ Ground truth** — الشهرة مرت بمرحلتين معاك

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | الشهرة مرت بمرحلتين معاك ✅ |
| Qwen3-ASR (base) | الشهرة مرت بمرحلتين **معك** ⚠️ |
| Cohere | الشهره مرت بمرحلتين معاك ✅ |
| Nemotron | الشهرة مرت بمرحلتين **معك** ⚠️ |
| Arabic-Whisper-CS | الشهرة مرت بمرحلتين معاك ✅ |
| MasriSwitch | الشهرة مرت بمرحلتين معاك ✅ |

**Medium**

> **✅ Ground truth** — لا ركز عشان انت دلوقتي هتتحاسب على الكلام ده

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | لا ركز عشان انت دلوقتي هتتحاسب على الكلام ده ✅ |
| Qwen3-ASR (base) | لا ركز عشان أنت دلوقتي **هتحسب** على **كلام** ده ⚠️ |
| Cohere | لا ركز عشان انت دلوقتي هتتحاسب **عالكلام** ده ⚠️ |
| Nemotron | لأ ركز **على شان** أنت **دي الوقت هتتحسب** على الكلام ده ❌ |
| Arabic-Whisper-CS | لأ ركز عشان انت **دلوقت هتتحسب** على الكلام ده ⚠️ |
| MasriSwitch | لأ ركز عشان انت دلوقتي هتتحاسب على الكلام ده ✅ |

> **✅ Ground truth** — بصراحة كانت من أسعد أيام حياتي لما شفت اللي هما فرحانين

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | بصراحة كانت من أسعد أيام حياتي لما شفت اللي هما فرحانين ✅ |
| Qwen3-ASR (base) | بصراحة **كنت** من أسعد أيام حياتي لما **شوفت** اللي **هم فرحاني** ❌ |
| Cohere | بصراحه كانت **قبل اسعد قيم** حياتي لما **شوفت** اللي هم فرحانين ❌ |
| Nemotron | بصراحة كانت من أسعد **قام حاتي** لما شفت اللي **هم فرحين** ❌ |
| Arabic-Whisper-CS | بصراحة كانت **أبنة** أسعد **أقام** حياتي لما **شوفت** اللي **هم** فرحانين ❌ |
| MasriSwitch | بصراحة كانت من أسعد أيام حياتي لما شفت اللي **هم** فرحانين ⚠️ |

**Long**

> **✅ Ground truth** — انا النهاردة مثلا ساكن في مدينة نصر وشغلي في المعادي فانا من البيت للشغل بقطع 20 كيلو في اليوم

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | انا النهارده مثلا ساكن في مدينة نصر وشغلي في المعادي فانا من البيت للشغل بقطع 20 كيلو في اليوم ✅ |
| Qwen3-ASR (base) | أنا النهاردة مثلاً ساكن في مدينة نصر وشغلي في المعادي فأنا من البيت للشغل **لبقيت** 20 كيلو في اليوم ⚠️ |
| Cohere | انا النهارده مثلا ساكن في مدينه **مصر** وشغلي في المعادي فانا من البيت للشغل بقطع 20 كيلو في اليوم ⚠️ |
| Nemotron | أنا النهار ده مثلاً ساكن في مدينة نصر وشغلي في المعادي فأنا من البيت للشغل بقطع 20 كيلو في اليوم ✅ |
| Arabic-Whisper-CS | أنا النهاردة مثلا ساكن في مدينة نصر و شغلي في المعادي فانا من البيت للشغل بقطع 20 كيلو في اليوم ✅ |
| MasriSwitch | أنا النهاردة مثلا ساكن في مدينة نصر وشغلي في المعادي فأنا من البيت للشغل بقطع 20 كيلو في اليوم ✅ |

> **✅ Ground truth** — استغلت كان فيه معاناة في تحضير المنهج

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | استغلت كان فيه معاناة في تحضير المنهج ✅ |
| Qwen3-ASR (base) | **اشتغلت** كان في معاناة في تحضير **المناك** ❌ |
| Cohere | **اشتغلت** كان في معاناه في تحضير **المنعكس** ❌ |
| Nemotron | آه **استقالت** كان في **معناه** في تحضير **المناك** ❌ |
| Arabic-Whisper-CS | **اشتغلت** كان في **معناه** في تحضير المنهج ⚠️ |
| MasriSwitch | **إشتغلت** كان في **معنية** في تحضير المنهج ❌ |

---

## 📦 Installation

> **Install the right torch first.** A plain `pip install` pulls the newest torch
> (built for the latest CUDA), which fails on older drivers with *"NVIDIA driver
> too old"*. Install a torch build matching **your** driver **before** the package,
> then add QwenCleo with `--no-deps` so torch is never reinstalled.
>
> Pick the wheel index for your CUDA driver — `cu121` (driver ≥ 12.1, e.g. CUDA 12.2),
> `cu118` (driver ≥ 11.8), or `cpu`. Check yours with `nvidia-smi`.

### For inference & chunked transcription (PyPI)

```bash
conda create -n qwencleo-asr python=3.12 -y
conda activate qwencleo-asr

# 1) torch matching your driver (cu121 shown — change the index for yours)
pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 2) QwenCleo without touching torch, then its remaining deps
pip install qwencleo-asr --no-deps
pip install "qwen-asr>=0.0.6" numpy soundfile huggingface_hub
```

That's all you need for the Python API and the `qwencleo` CLI.


### For serving / Gradio / vLLM (clone the repo)

```bash
conda create -n qwencleo-asr python=3.12 -y
conda activate qwencleo-asr

# 1) torch matching your driver, first
pip install torch==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 2) the repo (without re-resolving torch) + serving deps
git clone https://github.com/MohammedAly22/qwencleo-asr.git
cd qwencleo-asr
pip install -e . --no-deps
pip install "qwen-asr>=0.0.6" numpy soundfile huggingface_hub
pip install -r requirements-serving.txt
```

Verify torch sees the GPU before running:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.5.1+cu121 True
```

---

## 🚀 Usage

### Python — basic transcription

```python
from qwencleo_asr import QwenCleoASR

asr = QwenCleoASR()                       # loads mohammedaly22/QwenCleo-ASR
result = asr.transcribe("clip.wav")       # language defaults to "Arabic"
print(result.text)
```

Batch, auto-detect language, and Egyptian normalization:

```python
results = asr.transcribe(["a.wav", "b.wav"], language=None)   # auto-detect
clean   = asr.transcribe("clip.wav", normalize=True)          # normalized text
```

### Python — chunked transcription of long audio / mic

```python
from qwencleo_asr import QwenCleoASR, stream_file

asr = QwenCleoASR()
for chunk in stream_file(asr, "long_podcast.wav", chunk_s=20, overlap_s=2):
    print(f"[{chunk.start:.0f}-{chunk.end:.0f}s] {chunk.text}")
```

> ℹ️ **This is chunked transcription, not true streaming.** It splits long/live
> audio into overlapping windows and transcribes each — convenient for captioning
> without a server, but latency is per-window. For genuine **token-by-token
> streaming**, use the vLLM path below.

### Python — true streaming (vLLM)

QwenCleo inherits Qwen3-ASR's **real token-by-token streaming** via vLLM. Start a
server (see [`server/vllm_serve.md`](server/vllm_serve.md)):

```bash
pip install "qwencleo-asr[vllm]"          # vLLM nightly recommended — see docs
vllm serve mohammedaly22/QwenCleo-ASR
```

Then stream straight off the model object — deltas arrive as they're generated:

```python
from qwencleo_asr import QwenCleoASR

asr = QwenCleoASR()
for delta in asr.stream("clip.wav"):       # talks to the vLLM server
    print(delta, end="", flush=True)
```

Or use the helpers directly:

```python
from qwencleo_asr import stream_vllm, transcribe_vllm, VLLMOffline

for delta in stream_vllm("clip.wav", language="Arabic"):
    print(delta, end="", flush=True)

print(transcribe_vllm("clip.wav"))         # one-shot via the server
print(VLLMOffline().transcribe("clip.wav"))  # in-process, no server
```

### CLI

```bash
qwencleo transcribe clip.wav
qwencleo transcribe a.wav b.wav --language None --normalize
qwencleo stream long_podcast.wav --chunk-s 20 --overlap-s 2
```

---

## 🌐 Serving

### FastAPI server

```bash
QWENCLEO_MODEL=mohammedaly22/QwenCleo-ASR \
uvicorn server.app:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/v1/transcribe -F file=@clip.wav -F language=Arabic
```

### Gradio demo

```bash
python app/gradio_app.py        # http://localhost:7860  (mic + file upload)
```

### vLLM — serving, streaming & OpenAI-compatible API

Full guide in **[`server/vllm_serve.md`](server/vllm_serve.md)**. In short:

```bash
pip install "qwencleo-asr[vllm]"           # vLLM nightly recommended (see docs)
vllm serve mohammedaly22/QwenCleo-ASR
```

OpenAI-compatible transcription:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
print(client.audio.transcriptions.create(
    model="mohammedaly22/QwenCleo-ASR", file=open("clip.wav","rb").read()).text)
```

### Streaming mic web demo

Live browser-mic transcription via the upstream Flask demo:

```bash
qwen-asr-demo-streaming \
  --asr-model-path mohammedaly22/QwenCleo-ASR \
  --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.9
# open http://<your-ip>:8000
```

---

## 📓 Examples (Colab)

Runnable notebooks in [`examples/`](examples/) — open one, set the runtime to
**GPU** (*Runtime → Change runtime type → GPU*), and run the cells top to bottom.

| Notebook | What it shows | Open in Colab |
|---|---|:---:|
| **Quick Start** | Install, transcribe, batch, CLI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/qwencleo-asr/blob/main/examples/01_quickstart.ipynb) |
| **Chunked transcription** | Long-audio windowing + mic-style frames | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/qwencleo-asr/blob/main/examples/02_chunked_transcription.ipynb) |
| **FastAPI server** | Run the server, call it over HTTP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/qwencleo-asr/blob/main/examples/03_fastapi_server.ipynb) |
| **Gradio demo** | Browser UI with a public share link | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/qwencleo-asr/blob/main/examples/04_gradio_demo.ipynb) |
| **vLLM streaming** | True token-by-token streaming + OpenAI API | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/qwencleo-asr/blob/main/examples/05_vllm_streaming.ipynb) |

---

## 🔗 Links

- **🤗 Model card:** [`mohammedaly22/QwenCleo-ASR`](https://huggingface.co/mohammedaly22/QwenCleo-ASR)
- **📦 PyPI:** [`qwencleo-asr`](https://pypi.org/project/qwencleo-asr/)
- **🧱 Base model:** [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [Qwen3-ASR repo](https://github.com/QwenLM/Qwen3-ASR)
- **Languages:** Egyptian Arabic, Modern Standard Arabic, Arabic↔English code-switching
- **Recommended `language` hint:** `"Arabic"` (or `None` to auto-detect)

---

## 📜 License & citation

Apache-2.0, inheriting the Qwen3-ASR license terms.

```bibtex
@misc{qwencleo_asr_2026,
  title  = {QwenCleo-ASR: The Best Open-Source Egyptian Arabic and Code-Switching Speech Recognition Model},
  author = {Mohammed Aly},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/mohammedaly22/QwenCleo-ASR}},
  note   = {Fine-tuned from Qwen3-ASR-1.7B}
}
```
