# Skill-Detect — LiteLLM Gateway + Gemma QLoRA Skill Extraction

Fine-tune an open model with PEFT/LoRA to extract skills from resumes and job
descriptions, served behind a LiteLLM gateway. Base taxonomy is **ESCO**
(open); SkyHive/Clustree are proprietary, so they plug in only as
`third_party:*` namespaces mapped onto ESCO — plus your own `custom:*` skills
learned by LoRA without touching base weights.

Architecture diagram (open in a browser):
`.archify/architecture-skill-detect-20261011-120000/skill-detect.html`

Serving + monitoring: `docs/SERVING.md` (measured throughput, capacity,
Prometheus/Grafana). GPU train proof: `docs/GPU_PROOF.md`.

## Quickstart (Windows, D: drive)

```powershell
D:; cd D:\skill-project
python -m venv .venv --system-site-packages; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"          # light: tests only
pip install transformers peft trl datasets accelerate safetensors huggingface_hub  # train/infer
python -m pytest -q              # 33 passed, 1 skipped (regression needs train deps); no network/GPU
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

Keep model weights off C: (only ~8GB free there) — the project uses
`HF_HOME=D:\skill-project\.hf-cache` (persisted via `setx`; gitignored).

```powershell
.\.venv\Scripts\python -m pip install "numpy==1.26.4" torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 `
  --index-url https://download.pytorch.org/whl/cu126
```

Proof run — ungated tiny model, fp16 LoRA, GPU end-to-end in minutes:

```powershell
.\.venv\Scripts\python scripts\train_qlora.py --model HuggingFaceTB/SmolLM2-135M-Instruct `
  --max-steps 50 --quant none
.\.venv\Scripts\python scripts\eval_model.py --model HuggingFaceTB/SmolLM2-135M-Instruct `
  --adapter adapters\smollm2-proof
```

Proven on RTX 3070 (2026-10-11): 50 steps, 6.3 epochs, loss **3.76 → 2.67**,
1.84M trainable params (1.35%), adapter in `adapters/smollm2-proof`.
Real-generation F1 = 0.0 — the 135M model talks about the right skill but
emits prose, not strict JSON. That format gap is exactly what the Gemma 4B +
assistant-only-loss run closes; the loop itself is proven end-to-end.

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

Response-only loss is real and default on: `--assistant-only` masks prompt
tokens via the model's own chat template (`add_generation_prompt` on history
only — rendering the full chat appends a duplicate assistant header, verified
against SmolLM2). Non-matching rows fall back to full-sequence loss and are
counted, never silent. Measured: 45/45 rows masked, 10-step loss **0.93 →
0.77**. `--no-assistant-only` forces full-sequence loss for template-less base
models.

## Serving (measured § docs/SERVING.md)

```powershell
$env:SKILL_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"; $env:SKILL_ADAPTER="adapters/smollm2-proof"
.\.venv\Scripts\python -m uvicorn skill_project.serve:app --host 127.0.0.1 --port 8000
# POST /extract (x-correlation-id honored) · GET /health · GET /metrics (Prometheus)
# OpenAI-compatible: GET /v1/models · POST /v1/chat/completions (extraction via last user turn)
# docker compose -f deploy/docker-compose.yml up  # Prometheus :9090, Grafana :3000
```

Proven live: 200 with real GPU weights, correlation echo, audit row, judge
alert counter, cost $0.00 local. Breaker/guardrails/judge/audit covered by tests.

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
src/skill_project/      → taxonomy.py gateway.py dataset.py training.py serve.py
scripts/                → fetch_esco, build_custom_skills, build_dataset,
                          train_qlora (transformers Trainer + PEFT LoRA), eval_f1, eval_model, demo_extract
tests/                  → unit + integration + serve + CPU regression (fakes only at LLM edge)
deploy/                 → docker-compose (Prometheus+Grafana), dashboard JSON
docs/                   → GPU_PROOF.md (measured train proof), SERVING.md (throughput/capacity)
data/ adapters/ logs/   → gitignored runtime artifacts
```

## Git / upload

```powershell
git add -A; git commit -m "msg"; git remote add origin <url>; git push -u origin main
```

Set your identity first: `git config user.name "you"; git config user.email "you@x"`.
Secrets stay in env (never in yaml); `data/` and `adapters/` are gitignored.
