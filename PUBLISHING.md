# Publishing QwenCleo-ASR

Three targets: **HuggingFace model** (weights + card), **PyPI** (the pip package),
and the **GitHub repo** (serving code, charts, assets). Run these on the H100 box
where your credentials/checkpoint live. Do them in this order.

> ⚠️ Before anything: confirm the QwenCleo metrics in `benchmarks/results.json`,
> `README.md`, and `MODEL_CARD.md` (currently 19.85 / 10.64) match a fresh rescore
> of `results/finetuned_hyps.jsonl` with the fixed normalizer:
> ```bash
> python scripts/rescore_from_dump.py --dump results/finetuned_hyps.jsonl \
>   --name QwenCleo-ASR --out results/qwencleo.json
> ```
> If they changed, update the three files + re-run `make_charts.py` before pushing.

---

## 1. Push the model to HuggingFace

```bash
pip install -U huggingface_hub
huggingface-cli login            # paste a WRITE token from hf.co/settings/tokens
```

```bash
# create the repo (once)
huggingface-cli repo create QwenCleo-ASR --type model -y

# stage the fine-tuned checkpoint + the model card + assets
#   CKPT = your best checkpoint dir (must contain processor files; if not, run
#   scripts/patch_checkpoint.sh on it first)
CKPT=/home/workspace/m.aly/qwen-asr-ft/outputs/qwen3asr-eg-2xh100/checkpoint-9868

python - <<'PY'
import os, shutil
from huggingface_hub import HfApi, upload_folder
repo = "mohammedaly22/QwenCleo-ASR"
ckpt = os.environ.get("CKPT")
api = HfApi()

# 1) upload model weights + config + processor from the checkpoint
upload_folder(repo_id=repo, folder_path=ckpt, repo_type="model",
              ignore_patterns=["optimizer.pt", "*.tmp", "global_step*", "trainer_state.json"])

# 2) upload the model card as README.md (HF renders this as the card)
api.upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md",
                repo_id=repo, repo_type="model")

# 3) upload the assets (banner + charts) the card references
upload_folder(repo_id=repo, folder_path="assets", path_in_repo="assets",
              repo_type="model")
print("done:", repo)
PY
```

**Verify:** open `https://huggingface.co/mohammedaly22/QwenCleo-ASR`, check the card
renders with banner + charts, then smoke-test loading:

```bash
python -c "
import torch; from qwen_asr import Qwen3ASRModel
m = Qwen3ASRModel.from_pretrained('mohammedaly22/QwenCleo-ASR', dtype=torch.bfloat16, device_map='cuda:0')
print(m.transcribe(audio='test_audios/1.wav', language='Arabic')[0].text)
"
```

> The checkpoint must include processor files (`preprocessor_config.json`,
> `tokenizer*`, `chat_template.json`, …). If missing:
> `bash scripts/patch_checkpoint.sh $CKPT` first.

---

## 2. Publish the package to PyPI

```bash
pip install -U build twine
cd qwencleo-asr            # the package root (where pyproject.toml is)

# clean previous builds, then build sdist + wheel
rm -rf dist build *.egg-info
python -m build           # -> dist/qwencleo_asr-0.1.0.tar.gz + .whl

# sanity: metadata + README render check
twine check dist/*
```

Test on TestPyPI first (recommended):

```bash
twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ qwencleo-asr   # in a clean venv
```

Then the real upload:

```bash
twine upload dist/*       # prompts for __token__ + a pypi.org API token
```

**Verify:** `pip install qwencleo-asr` in a fresh venv, then
`python -c "from qwencleo_asr import QwenCleoASR; print('ok')"`.

> The README banner/charts use absolute `raw.githubusercontent.com/.../main/...`
> URLs so they render on the PyPI page (PyPI can't show relative image paths).
> They only resolve after step 3 pushes the repo.

---

## 3. Push the code to GitHub

```bash
cd qwencleo-asr
git init -b main
git add .
git commit -m "QwenCleo-ASR: Egyptian Arabic & code-switching ASR (initial release)"
git remote add origin https://github.com/mohammedaly22/qwencleo-asr.git
git push -u origin main
```

**Verify:** the README renders on GitHub with banner + charts (these come from the
pushed `assets/` via the raw URLs), and the links to PyPI / HF resolve.

---

## Post-publish checklist

- [ ] HF card shows banner, results table, charts, sample outputs.
- [ ] `pip install qwencleo-asr` works in a clean venv.
- [ ] `from qwencleo_asr import QwenCleoASR; QwenCleoASR().transcribe(...)` works
      end-to-end against the published HF model.
- [ ] GitHub README images load (raw URLs resolve after push).
- [ ] PyPI page renders (may take a minute; images via raw GitHub URLs).
