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
  <a href="https://github.com/MohammedAly22/qwencleo-asr"><img src="https://img.shields.io/badge/GitHub-qwencleo--asr-181717?logo=github&logoColor=white" alt="GitHub"/></a>
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

QwenCleo inherits Qwen3-ASR's **true token-by-token streaming** via vLLM
(needs an Ampere-or-newer GPU: L4 / A100 / H100).

```bash
pip install qwencleo-asr
qwencleo install-vllm     # vLLM nightly (cu129) — only build with Qwen3-ASR support
qwencleo serve            # OpenAI-compatible server on :8000
```

```python
from qwencleo_asr import QwenCleoASR

asr = QwenCleoASR()
for delta in asr.stream("clip.wav", port=8000):   # real streaming via vLLM
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

---

## 🗣️ Sample outputs

Real transcriptions from the test set, scored after Egyptian-aware
normalization. Examples span **short, medium, and long** clips in both
**Egyptian Arabic** and **code-switching**. These are picked to be honest — on
some clips other models also do well; QwenCleo's edge is consistency, dialect
fidelity, and keeping English terms in Latin script.

> ✅ = matches ground truth · ⚠️ = minor slips · ❌ = clear errors. **Bold** marks the wrong spans.

### 🔀 Code-switching

**Short**

> **✅ Ground truth** ااه بحس ستايله حلو وشكله حلو كاريزما وهو بيلعب 

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | بحس **style** ه حلو وشكله حلو كاريزما وهو بيلعب ⚠️ |
| Qwen3-ASR (base) | أه بحس **طايله** حلو وشكله حلو و**كرزمه** وهو يلعب ❌ |
| Cohere | **اااااااا** بحس ستايله حلو وشكله حلو وكاريزما وهو بيلعب ✅ |
| Nemotron | آه بحس ستايله حلو وشكله حلو كاريزما هو بيلعب ✅ |
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
| Arabic-Whisper-CS | فبتديك كل ال `soft skills` بشكل `indirect` حرفيا بتديهالك بالمعلقة ليه بتتعلم تشتغل على كل الحاجات بتاعة `Microsoft, Excel, PowerPoint, Word whatever` بعدين كمان بتتعلم أن أنت ⚠️ |
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
| Cohere | لا ركز عشان انت دلوقتي هتتحاسب **عالكلام** ده ✅ |
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
| MasriSwitch | بصراحة كانت من أسعد أيام حياتي لما شفت اللي **هم** فرحانين ✅ |

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

> **✅ Ground truth** — اشتغلت كان فيه معاناة في تحضير المنهج

| Model | Output |
|---|---|
| 🥇 **QwenCleo** | اشتغلت كان فيه معاناة في تحضير المنهج ✅ |
| Qwen3-ASR (base) | **اشتغلت** كان في معاناة في تحضير **المناك** ❌ |
| Cohere | **اشتغلت** كان في معاناه في تحضير **المنعكس** ❌ |
| Nemotron | آه **استقالت** كان في **معناه** في تحضير **المناك** ❌ |
| Arabic-Whisper-CS | **اشتغلت** كان في **معناه** في تحضير المنهج ✅ |
| MasriSwitch | **إشتغلت** كان في **معنية** في تحضير المنهج ❌ |


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

- **💻 GitHub:** [`MohammedAly22/qwencleo-asr`](https://github.com/MohammedAly22/qwencleo-asr) — package source, server, Gradio app, vLLM streaming guide.
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
