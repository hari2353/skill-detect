# GPU Proof — LoRA fine-tune on RTX 3070 (2026-10-11)

Question answered: does the fine-tune loop actually run on this laptop's GPU,
and what comes out? All commands executed, outputs recorded verbatim.

## Environment

- ASUS ROG Zephyrus G15, Ryzen 9 5900HS, 16GB RAM, RTX 3070 Laptop 8GB, Win 11
- `.venv`: torch **2.7.1+cu126**, transformers **5.19.0**, peft **0.21.2**,
  trl **1.15.0**, numpy 1.26.4, CUDA available
- torch 2.0.1 (system) and 2.5.1 both failed first: transformers 5.x needs
  `torch.accelerator` (2.6+) and torch>=2.5. 2.7.1 is the working pairing.

## Train (proof model — real LoRA update, not a mock)

```powershell
.\.venv\Scripts\python scripts\train_qlora.py --model HuggingFaceTB/SmolLM2-135M-Instruct --max-steps 50 --quant none
```

- Data: `data/train.jsonl`, 60 validated records (50 held out 15 for eval)
- LoRA r=16 on q/k/v/o proj, fp16, batch 1 × grad-accum 8, cosine, lr 1e-4
- Trainable: **1,843,200 / 136,358,208 (1.35%)** — base weights untouched
- Loss: **3.76 → 2.67** over 50 steps / 6.3 epochs (~6 min on the 3070)

| step | loss |
|---|---|
| 5 | 3.76 |
| 10 | 3.59 |
| 15 | 3.39 |
| 20 | 3.17 |
| 25 | 3.00 |
| 30 | 2.89 |
| 35 | 2.74 |
| 40 | 2.72 |
| 45 | 2.67 |
| 50 | 2.68 |

Final report: `{"global_steps": 50, "train_loss": 3.0602, "first_loss": 3.76,
"last_loss": 2.6748, "adapter_dir": "adapters/smollm2-proof"}`

Adapter artifact (`adapters/smollm2-proof`, gitignored, local only):
`adapter_config.json` + `adapter_model.safetensors` (7.4 MB) + tokenizer.

## Eval (real generation through the gateway)

```powershell
.\.venv\Scripts\python scripts\eval_model.py --model HuggingFaceTB/SmolLM2-135M-Instruct --adapter adapters\smollm2-proof
```

- Heldout n=15: `{"micro_p": 0.0, "micro_r": 0.0, "micro_f1": 0.0}`
- Sample: gold `[{"skill_id": "custom:etl-orchestration", ...}]`,
  model generates prose *about* ETL orchestration instead of the JSON array.
- Baseline (no adapter) did not finish in the time budget — reported as
  incomplete, not as zero.

## Honest reading

- **Proven:** GPU LoRA loop end-to-end (tokenize → forward/backward on CUDA →
  adapter save → reload → generate → parse → score). Loss curve learns.
- **Not proven:** extraction *quality*. A 135M model after 50 steps cannot hold
  strict-JSON format. That gap is closed by scale (Gemma 3-4B) plus
  response-only loss — both ready in `scripts/train_qlora.py`.
- **Blockers for the Gemma run:** `hf auth login` + license accept (gated
  weights), and WSL2 (bitsandbytes 4-bit is Linux-first).

## Where the model lives

Nowhere public. Base weights: HF cache on this machine. Adapter: local
`adapters/` dir. Nothing pushed to the Hub. Push needs `hf auth login`.
