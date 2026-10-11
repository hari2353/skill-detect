# Plan: Skill-Detect (implements SPEC.md)

## Components & dependencies
1. `skill-data` (taxonomy + dataset builder) — no deps. FIRST.
2. `gateway` (LiteLLM router + extractor) — depends on `skill-data` types only.
3. `finetune` (QLoRA config + train script) — depends on `skill-data` records.
4. `serve` (local endpoint + litellm.yaml wiring) — depends on `gateway`, `finetune`.

## Order & parallelism
- Sequential: data contracts → gateway/finetune → serve (each slice demoable).
- Parallelizable later: encoder NER track, Lightcast/O*NET loaders, eval dashboard.

## Risks (RTX 3070 8GB, 16GB RAM)
- OOM on 7B → mitigated: default 3B + 4-bit + grad-checkpoint + batch 1 (see lora.yaml).
- Old torch 2.0.1+cu117 on system → mitigated: isolated `.venv`, `train` extra pins.
- Disk: keep everything on D: (288GB free); F: untouched.
- bitsandbytes on Windows → mitigated: WSL2 recommended for real train; `--smoke`
  path stays native-Windows.

## Verification checkpoints
- [x] Slice 1: scaffold + unit tests green (this change)
- [ ] Slice 2: ESCO snapshot ingest + 200-record public/synthetic dataset validates
- [ ] Slice 3: `--smoke` adapter-graph build on CPU
- [ ] Slice 4 (opt-in GPU): 3B QLoRA short run + eval F1 + LiteLLM live route demo
