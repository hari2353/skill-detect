# Spec: Skill-Detect — LiteLLM Gateway + QLoRA Skill Extraction

## Objective
Prove that an open-source model can be PEFT/LoRA fine-tuned for **skill detection**
(resume/JD text → structured skills) on a laptop GPU, served behind a **LiteLLM
gateway** that routes across base taxonomy, third-party and custom skills.

User stories:
- As a builder, I send `POST /extract {text}` and get `[{id, label, namespace, span}]`
  with ESCO IDs for known skills and `custom:*` IDs for my own skills.
- As a builder, I can send a standard `POST /v1/chat/completions` (OpenAI shape)
  and get the hits back as the assistant message — so any OpenAI SDK/client works
  unchanged. `GET /v1/models` lists what's served.
- As a builder, I register a custom skill (label + 5 example phrases) and it is
  detectable after a short LoRA re-train, without touching base weights.
- As a builder, I can swap the backing model (local QLoRA endpoint ↔ commercial
  fallback) via LiteLLM config only — no code change.

Success = `tests/` green + `train --smoke` builds adapters on CPU + eval F1
reported on a held-out slice. Full 3B QLoRA run is opt-in (needs ~15GB free).

## Tech Stack
- Python 3.11, `litellm>=1.50` (gateway), `transformers`, `peft`, `datasets`,
  `pyyaml`, `pytest`. Serving: `fastapi`, `uvicorn`, `prometheus_client`.
- Trainer is plain `transformers.Trainer` + PEFT LoRA (response-only loss).
  TRL/bitsandbytes are Linux-first (triton/4-bit) — used in WSL2, not on native
  Windows; see `scripts/train_qlora.py` header for why.
- Base model (default): `google/gemma-3-4b-it` — strongest strict-JSON extraction
  in the 8GB QLoRA class (Unsloth required). Alt: `Qwen2.5-3B-Instruct` (lighter,
  comfortable without Unsloth). 7B+ documented but NOT default.
- Base taxonomy: **ESCO** (EU, free; snapshot as JSONL). Extension point for
  Lightcast Open Skills / O*NET via same `SkillTaxonomy` protocol.
- SkyHive/Clustree are proprietary (Cornerstone) — supported only as
  `third_party:<vendor>` namespaces mapped onto ESCO IDs, never as base.

## Commands
```powershell
D:; cd D:\skill-project
python -m venv .venv --system-site-packages; .\.venv\Scripts\Activate.ps1
pip install transformers peft trl datasets accelerate safetensors huggingface_hub  # train/infer
python -m pytest -q                           # Test (24 tests, no network/GPU)
python scripts\build_custom_skills.py --help  # synthetic custom-skill data
python scripts\fetch_esco.py --offline --limit 500  # ESCO snapshot (seed fallback offline)
python scripts\build_dataset.py --help              # train/heldout JSONL, validated
python scripts\demo_extract.py                      # offline end-to-end demo (no keys)
python scripts\eval_f1.py --heldout data\heldout.jsonl --mode oracle  # self-test scorer
python scripts\train_qlora.py --help                # real TRL SFT (proof: SmolLM2-135M)
python scripts\eval_model.py --help                 # score local base+adapter on heldout
python -m skill_project.training --smoke      # smoke: config + prompt wiring, no download
```

## Project Structure
```
D:\skill-project\
  SPEC.md                    → this file (living doc)
  tasks\plan.md tasks\todo.md → plan + task list
  configs\litellm.yaml       → gateway router (local adapter → fallback)
  configs\lora.yaml          → QLoRA hyperparams for 8GB VRAM
  src\skill_project\
    taxonomy.py              → SkillTaxonomy protocol, composite, custom registry
    gateway.py               → CompletionClient protocol, LiteLLM client, router
    dataset.py               → instruction-tuning record builder + validation
    training.py              → LoRA config + prompt builder + guarded train()
  scripts\build_custom_skills.py → synthetic third-party/custom data generator
  tests\                     → unit tests with fakes (no network/GPU)
  data\                      → gitignored; ESCO snapshots + generated JSONL land here
```

## Code Style
SOLID + thin slices. Depend on protocols, not concretions:
```python
def extract_skills(text: str, extractor: SkillExtractor) -> list[SkillHit]:
    return extractor.extract(text)  # caller never imports litellm
```
- `snake_case` funcs, `PascalCase` types, `frozen` dataclasses for value objects.
- One responsibility per module; new taxonomy/vendor = new class, never an edit
  to `CompositeTaxonomy` (Open/Closed).

## Testing Strategy
- `pytest`, tests in `tests\`, fakes for LLM/taxonomy (no network, no GPU).
- Coverage expectation: `taxonomy.py`, `gateway.py`, `dataset.py` fully covered.
- Levels: unit (now) → smoke `--smoke` (adapter graph on CPU) → GPU train + eval
  F1 (opt-in, manual checkpoint).

## Boundaries
- Always: run `pytest` before any train change; validate dataset records before train;
  keep secrets in env (never in yaml).
- Ask first: changing base model, adding heavy deps, touching partition/disk setup.
- Never: commit API keys, commit `data\` or `adapters\`, full-finetune base weights,
  repartition disks.

## Success Criteria
- [ ] `pytest -q` green on a fresh clone with only `[dev]` extras
- [ ] `SkillRouter` demonstrably tries custom-adapter → base → fallback (test)
- [ ] `dataset.build_record` rejects records with unknown skill IDs (test)
- [ ] `--smoke` completes on CPU without downloading a model
- [ ] Docs show the exact GPU QLoRA command + VRAM expectations for RTX 3070 8GB

## Open Questions
1. ESCO vs Lightcast as primary IDs long-term? (Default: ESCO; Lightcast via mapping.)
2. Encoder NER track later? (Noted in plan as Phase 2; generative proof first.)
