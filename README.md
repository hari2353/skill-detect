# Skill-Detect — LiteLLM + QLoRA proof (runs on RTX 3070 8GB)

Base taxonomy **ESCO** (open). SkyHive/Clustree are proprietary → `third_party:*`
namespaces mapped onto ESCO, plus `custom:*` skills learned via LoRA.

## Quickstart (Windows, D: drive)
```powershell
D:; cd D:\skill-project
py -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
python scripts\fetch_esco.py --offline --limit 60
python scripts\build_custom_skills.py --n 60
python scripts\build_dataset.py --esco-n 60
python scripts\demo_extract.py
```

## Slices
1. Scaffold + unit tests (you are here).
2. `python scripts/fetch_esco.py --limit 500` → `data/esco_skills.jsonl` (ingest).
3. `python scripts/build_custom_skills.py --out data/custom_train.jsonl --n 200`.
4. CPU smoke: `python -m skill_project.training --smoke`.
5. GPU (opt-in, WSL2 recommended): see `configs/lora.yaml` + SPEC.md.

## VRAM reality (RTX 3070 8GB)
| Model | Mode | Fits? |
|---|---|---|
| Gemma 3-4B-it (default) | QLoRA 4-bit + Unsloth, r=16, batch 1, grad-ckpt | yes, Unsloth mandatory |
| Qwen2.5-3B / Llama-3.2-3B | QLoRA 4-bit, r=16, batch 1, grad-ckpt | yes, comfortable |
| Mistral-7B / Llama-3.1-8B | QLoRA 4-bit + Unsloth, batch 1 | edge, slow |
| Any 3B+ | full finetune | no |

## SOLID map
- `SkillTaxonomy` protocol → `InMemoryTaxonomy`, `CustomSkillRegistry`, future
  `EscoApiTaxonomy` / `LightcastTaxonomy` (add classes, never edit composite).
- `CompletionClient` protocol → `FakeClient` (tests), `LiteLLMClient` (prod).
- `SkillRouter`: custom-adapter → base → fallback. Order is config, not code.
