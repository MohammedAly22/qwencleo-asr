---
license: apache-2.0
language:
- ar
library_name: qwen-asr
pipeline_tag: automatic-speech-recognition
base_model: Qwen/Qwen3-ASR-1.7B
tags:
- automatic-speech-recognition
- speech
- arabic
- egyptian-arabic
- code-switching
- asr
- qwen3-asr
metrics:
- wer
- cer
model-index:
- name: QwenCleo-ASR
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: Egyptian Arabic + Code-Switching test set
      type: custom
    metrics:
    - type: wer
      value: 19.85
      name: WER (overall)
    - type: cer
      value: 10.64
      name: CER (overall)
---

<p align="center">
  <img src="assets/QwenCleo-ASR-Banner.png" alt="QwenCleo-ASR" width="100%"/>
</p>

<h1 align="center">🎙️ QwenCleo-ASR</h1>

<h3 align="center">The best open-source model for Egyptian Arabic &amp; code-switching speech recognition</h3>

<p align="center"><em>Built on <a href="https://huggingface.co/Qwen/Qwen3-ASR-1.7B">Qwen3-ASR-1.7B</a>, fine-tuned for Egyptian dialect and Arabic ↔ English code-switching.</em></p>

<p align="center">
  <a href="https://pypi.org/project/qwencleo-asr/"><img src="https://img.shields.io/badge/PyPI-qwencleo--asr-3775A9?logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="https://huggingface.co/Qwen/Qwen3-ASR-1.7B"><img src="https://img.shields.io/badge/Base-Qwen3--ASR--1.7B-615CED" alt="Base model"/></a>
</p>

---

<p align="center">
  <b>QwenCleo</b> — the name carries three meanings: <b>Qwen</b>, the powerful base model it is built on;
  <b>Queen</b>, signalling a model that reigns over its domain; and <b>Cleo</b>, for <b>Cleopatra</b>,
  the queen of Egypt — because this model is tailored for <b>Egyptian</b> Arabic. 👑🏺
</p>

<p align="center">
  <b>QwenCleo-ASR is, to our knowledge, the best open-source ASR model for Egyptian Arabic and
  Arabic/English code-switching.</b> It cuts the error rate of the strong Qwen3-ASR base
  <b>roughly in half</b>, and correctly keeps embedded English tech/loan words in Latin script
  (<code>engineering</code>, <code>download</code>, <code>React</code>, <code>at least</code>)
  instead of mangling them into broken Arabic.
</p>

---

## 🚀 Quick start

```bash
pip install qwencleo-asr
```

```python
from qwencleo_asr import QwenCleoASR

asr = QwenCleoASR()                       # loads mohammedaly22/QwenCleo-ASR
print(asr.transcribe("clip.wav").text)
```

Or with the upstream `qwen-asr` API directly:

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "mohammedaly22/QwenCleo-ASR", dtype=torch.bfloat16, device_map="cuda:0",
)
result = model.transcribe(audio="clip.wav", language="Arabic")
print(result[0].text)
```

---

## ⚡ Streaming & serving (vLLM)

QwenCleo inherits Qwen3-ASR's **true token-by-token streaming** via vLLM. Start a
server, then stream straight off the model:

```bash
pip install "qwencleo-asr[vllm]"          # vLLM nightly recommended
vllm serve mohammedaly22/QwenCleo-ASR
```

```python
from qwencleo_asr import QwenCleoASR

asr = QwenCleoASR()
for delta in asr.stream("clip.wav"):       # real streaming via the vLLM server
    print(delta, end="", flush=True)
```

OpenAI-compatible transcription, a live **mic web demo**
(`qwen-asr-demo-streaming`), offline `VLLMOffline`, and cURL examples are in the
[vLLM serving guide](https://github.com/MohammedAly22/qwencleo-asr/blob/main/server/vllm_serve.md).

> For captioning long files **without** a server, `stream_file(...)` does chunked
> (windowed) transcription — convenient, but latency is per-window, not per-token.

---

<h2 align="center">📊 Results</h2>

<p align="center">
  WER / CER (%) on an Egyptian-Arabic + code-switching test set (3,699 utterances).<br/>
  <b>Lower is better.</b> All models scored with the same Egyptian-aware normalization.
</p>

<p align="center">
  <img src="assets/QwenCleo-ASR-Benchmark.png" alt="Benchmark overview" width="100%"/>
</p>

<div align="center">

| Model | Params | WER all | CER all | WER · AR | CER · AR | WER · CS | CER · CS |
|---|---:|---:|---:|---:|---:|---:|---:|
| **🏆 QwenCleo-ASR** | 1.7B | **19.85** | **10.64** | **19.08** | **10.43** | **20.29** | **10.92** |
| NVIDIA Nemotron-3.5 | 0.6B | 38.88 | 20.58 | 37.14 | 17.40 | 42.15 | 26.30 |
| Qwen3-ASR-1.7B (base) | 1.7B | 41.51 | 20.86 | 40.59 | 18.52 | 43.20 | 25.04 |
| Whisper Large-v3 Turbo (FT) | 0.81B | 50.83 | 22.86 | 48.37 | 18.42 | 55.08 | 37.84 |
| Cohere Transcribe 03-2026 | 2.0B | 53.78 | 39.63 | 48.57 | 34.12 | 63.76 | 49.66 |
| Whisper Large-v3 | 1.54B | 63.94 | 39.76 | 49.25 | 22.76 | 59.32 | 31.52 |
| Whisper Large-v2 | 1.54B | 72.34 | 48.73 | 60.75 | 33.21 | 66.85 | 40.75 |
| Whisper Large-v3 Turbo | 0.81B | 73.83 | 46.86 | 59.37 | 29.42 | 66.08 | 37.84 |
| Whisper Medium | 0.76B | 80.46 | 53.19 | 74.77 | 41.76 | 74.15 | 44.90 |
| Whisper Small | 0.24B | 89.99 | 60.34 | 77.42 | 42.53 | 87.09 | 55.22 |
| Whisper Tiny | 0.04B | 124.68 | 89.42 | 116.02 | 77.74 | 110.67 | 74.57 |

</div>

---

## 🗣️ Sample outputs

Real transcriptions from the test set. **Ground truth** first; each model's output below it.
QwenCleo keeps English terms in Latin script and Egyptian dialect intact, while the other
models transliterate English into broken Arabic or drop words entirely.

### 🔀 Code-switching

> **✅ Ground truth** — طب وانتوا يعني ك`engineering` المفروض ان بيكون مثلا ال`staff engineer` بيقعد مع ال`engineering managers`

| Model | Output |
|---|---|
| 🏆 **QwenCleo** | طب وانتوا يعني ك`engineering` المفروض ان بيكون مثلا ال`staff engineer` بيقعد مع ال`engineering managers` ✅ |
| Qwen3-ASR (base) | وأنتوا يعني كإنجنييرينج المفروض إنه بيكون مثلاً الأستاف إنجنيير ❌ |
| Cohere | وانتم كانجينيري المفروض ان يكون مثلا الاستفاده من الهدف ❌ |
| Nemotron | وانتو يعني كإنجينير المفروض ان بيكون مثلاً الإستف إنجنير ❌ |

> **✅ Ground truth** — يعني شوية حاجات كده `across` كل ال`domains` او `at least` يعني مع 4 5 `squads` فالموضوع صعب

| Model | Output |
|---|---|
| 🏆 **QwenCleo** | يعني شوية حاجات كده `across` كل ال`domains` او `at least` يعني مع 4 5 `squads` فالموضوع صعب ✅ |
| Qwen3-ASR (base) | يعني شوية حاجات كده أكرس كل الدومينز وأتلست يعني مع أربعة خمسة سكوات ❌ |
| Cohere | يعني شويه حاجات كده اكروس كل الدومينز او اتليست مع اربع خمسه سكوات ❌ |
| Nemotron | يعني آه شوائد حاجات كده أكرس كل الدومين أو أتليست ❌ |

### 🇪🇬 Egyptian Arabic

> **✅ Ground truth** — طب دي كانت مثلا تاخد 84% 88%

| Model | Output |
|---|---|
| 🏆 **QwenCleo** | طب دي كانت مثلا تاخد 84% 88% ✅ |
| Qwen3-ASR (base) | طب دي كانت مثلاً تأخذ أربعة وثمانين في المية، ثمانية وثمانين في المية ❌ |
| Cohere | طيب دي كانت مثلا تاخد اربعه وثمانين في الميه ثمانيه وثمانين في الميه ❌ |
| Nemotron | طيب دي كانت مثلاً تاخد أربعة وثمانين في المئة ثمانية وثمانين في المئة ❌ |

> **✅ Ground truth** — خد ال 4 في 4 او 4 ونص طلع دور 9

| Model | Output |
|---|---|
| 🏆 **QwenCleo** | خد ال 4 في 4 او 4 ونص طلع دور 9 ✅ |
| Qwen3-ASR (base) | خادل الأربعة فاربعة واربعة ونص تطلع دور تاسع ❌ |
| Cohere | خد الاربعه فاربعه واربعه ونص طلع دور تسعه ❌ |
| Nemotron | خد الأربعة في أربعة وأربعة ونص طلع دور تسعة ❌ |

---

## 🧠 Model details

- **Base model:** [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) ([repo](https://github.com/QwenLM/Qwen3-ASR))
- **Fine-tuning data:** hundreds of hours of Egyptian podcast speech (Egyptian Arabic + Arabic/English code-switching).
- **Languages:** Egyptian Arabic, Modern Standard Arabic, Arabic ↔ English code-switching.
- **Recommended `language` hint:** `"Arabic"` (or `None` to auto-detect).
- **Intended use:** transcription/captioning of Egyptian Arabic and code-switched speech.
- **Limitations:** tuned for Egyptian dialect; other Arabic dialects and noisy far-field audio may degrade. Not optimized for pure-English audio.

---

## 🔗 Links

- **📦 PyPI:** [`qwencleo-asr`](https://pypi.org/project/qwencleo-asr/)
- **🧱 Base:** [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [Qwen3-ASR repo](https://github.com/QwenLM/Qwen3-ASR)

## 📜 Citation

```bibtex
@misc{qwencleo_asr_2026,
  title  = {QwenCleo-ASR: The Best Open-Source Egyptian Arabic and Code-Switching Speech Recognition Model},
  author = {Mohammed Aly},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/mohammedaly22/QwenCleo-ASR}},
  note   = {Fine-tuned from Qwen3-ASR-1.7B}
}
```

*License: Apache-2.0, inheriting the Qwen3-ASR license terms.*
