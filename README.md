# Skill-Detect — LiteLLM Gateway + Gemma QLoRA Skill Extraction

Fine-tune an open model with PEFT/LoRA to extract skills from resumes and job
descriptions, served behind a LiteLLM gateway. Base taxonomy is **ESCO**
(open); SkyHive/Clustree are proprietary, so they plug in only as
`third_party:*` namespaces mapped onto ESCO — plus your own `custom:*` skills
learned by LoRA without touching base weights.

Architecture diagram (open in a browser):
`.archify/architecture-skill-detect-20261011-120000/skill-detect.html`

## Quickstart (Windows, D: drive)

```powershell
D:; cd D:\skill-project
python -m venv .venv --system-site-packages; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"          # light: tests only
pip install transformers peft trl datasets accelerate safetensors huggingface_hub  # train/infer
python -m pytest -q              # 24 tests, no network/GPU needed
```

Full pipeline (all offline, ~1 min):

```powershell
$env:PYTHONPATH="D:\skill-project\src"
python scripts\fetch_esco.py --offline --limit 60
python scripts\build_custom_skills.py --n 60
python scripts\build_dataset.py --esco-n 60     # train=60 heldout=15 dropped=0
python scripts\demo_extract.py                  # resume -> ESCO + custom hits
python scripts\eval_f1.py --mode oracle
python -m skill_project.training --smoke
```

## GPU training (RTX 3070 8GB, native Windows)

The `.venv` uses `--system-site-packages` to reuse the system torch, then
upgrades it (system torch 2.0.1 is too old for transformers 5.x):

```powershell
.\.venv\Scripts\python -m pip install "numpy==1.26.4" torch torchvision torchaudio `
  --index-url https://download.pytorch.org/whl/cu121
```

Proof run — ungated tiny model, fp16 LoRA, GPU end-to-end in minutes:

```powershell
.\.venv\Scripts\python scripts\train_qlora.py --model HuggingFaceTB/SmolLM2-135M-Instruct `
  --max-steps 50 --quant none
.\.venv\Scripts\python scripts\eval_model.py --model HuggingFaceTB/SmolLM2-135M-Instruct `
  --adapter adapters\smollm2-proof
```

Target run — Gemma 3-4B QLoRA (needs two things this laptop can't fake):

1. **License gate:** `hf auth login` + accept `google/gemma-3-4b-it` on HuggingFace.
2. **4-bit on Windows:** bitsandbytes is Linux-first. Either run the same
   `train_qlora.py --model google/gemma-3-4b-it --quant 4bit` inside **WSL2**,
   or install Unsloth there (mandatory at 4B/8GB, batch 1 + grad-checkpoint).

```powershell
# WSL2 (or once bitsandbytes works locally):
python scripts/train_qlora.py --model google/gemma-3-4b-it --quant 4bit --max-steps 300
python scripts/eval_model.py --model google/gemma-3-4b-it --adapter adapters/gemma3-4b-qlora
```

Response-only loss is on by default (`assistant_only_loss=True`); if a model's
chat template lacks generation markers, retry with `--no-assistant-only`.

## VRAM reality (RTX 3070 8GB)

| Model | Mode | Fits? |
|---|---|---|
| SmolLM2-135M (proof) | fp16 LoRA | yes, easily |
| Gemma 3-4B-it (default) | QLoRA 4-bit + Unsloth, r=16, batch 1, grad-ckpt | yes, Unsloth mandatory |
| Qwen2.5-3B / Llama-3.2-3B | QLoRA 4-bit, r=16, batch 1, grad-ckpt | yes, comfortable |
| Mistral-7B / Llama-3.1-8B | QLoRA 4-bit + Unsloth, batch 1 | edge, slow |
| Any 3B+ | full finetune | no |

## SOLID map

- `SkillTaxonomy` protocol → `InMemoryTaxonomy`, `CustomSkillRegistry`,
  future `EscoApiTaxonomy`/`LightcastTaxonomy` (add classes, never edit composite).
- `CompletionClient` protocol → `FakeClient` (tests), `LiteLLMClient` (cloud),
  `HFClient` (local base+adapter — same pipeline, one-constructor swap).
- `SkillRouter`: custom-adapter → base → fallback. Order is config, not code.

## Repo layout

```
SPEC.md tasks/          → living spec + plan + task list
configs/                → litellm.yaml (router), lora.yaml (Gemma QLoRA recipe)
src/skill_project/      → taxonomy.py gateway.py dataset.py training.py
scripts/                → fetch_esco, build_custom_skills, build_dataset,
                          train_qlora (TRL SFT), eval_f1, eval_model, demo_extract
tests/                  → unit + integration (fakes only at the LLM edge)
data/ adapters/         → gitignored runtime artifacts
```

## Git / upload

```powershell
git add -A; git commit -m "msg"; git remote add origin <url>; git push -u origin main
```

Set your identity first: `git config user.name "you"; git config user.email "you@x"`.
Secrets stay in env (never in yaml); `data/` and `adapters/` are gitignored.
