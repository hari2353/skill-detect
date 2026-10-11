# Serving, Throughput, Monitoring — measured on RTX 3070 8GB (2026-10-11)

## What serves what (honest map)

| Layer | Status | Notes |
|---|---|---|
| FastAPI `src/skill_project/serve.py` | **live-tested** | `/extract`, `/health`, `/metrics`, breaker, guardrails, judge, audit |
| Local weights via `HFClient` | **live-tested** | SmolLM2-135M + LoRA adapter, real GPU generation |
| LiteLLM router | configured | `configs/litellm.yaml`: local endpoint → commercial fallback |
| vLLM | **not on Windows** | Linux-only; compose stub ready for WSL2/Docker (`deploy/`) |
| Prometheus + Grafana | compose ready | `docker compose -f deploy/docker-compose.yml up` (needs Docker) |

## Measured numbers (SmolLM2-135M + adapter, fp16, RTX 3070)

- Generation: **~4.7 tok/s** median (HF pipeline, WDDM overhead included)
- Model resident: **285 MB** VRAM; context window **8192** (from model config)
- Live `/extract` (256 max tokens): first request 74 s (pipeline build incl.),
  steady state ~30–40 s → **~1.5 req/min sequential**
- Our prompts are tiny (~25 in-tokens), so context headroom is enormous;
  throughput is bound by per-token speed, not window.
- Cost accounting: local model = **$0.00** (`skill_cost_usd_total` stays 0);
  fallback model priced at gpt-4o-mini rates when used.

## Capacity guide

| Setup | Fits? | Throughput shape |
|---|---|---|
| 135M fp16, 8GB VRAM | easily (285 MB) | ~5 tok/s, sequential worker |
| Gemma 3-4B 4-bit, 8GB VRAM | yes (~2.5–3 GB + ctx) | est. 15–25 tok/s (measure in WSL2) |
| 135M fp16, **4 GB VRAM** | yes, + room for parallel workers | same tok/s, can run N copies |
| Gemma 4B 4-bit, **4 GB VRAM** | tight, short context only | Unsloth/vLLM + small max-tokens |
| 135M on CPU, 4 GB RAM | yes (~0.5–1 GB RSS) | ~2–5 tok/s CPU, fine for dev |
| Parallel requests | semaphore `max_inflight=4`, 503 when full | real batching needs vLLM (Linux) |

Scale knobs: `SKILL_MAX_INFLIGHT`, uvicorn `--workers N` (each holds a model
copy — cheap for 135M, not for 4B), then vLLM continuous batching on Linux.

## Resilience in place

- **Circuit breaker** (`failure_threshold=3`, `cooldown_s=30`): opens on repeated
  model failure → 503 + `skill_circuit_state=1`; half-open probe after cooldown.
  Live-tested (test forces open state).
- **Guardrails**: unknown IDs dropped, ungrounded spans dropped, max 20 hits;
  every drop counted in `skill_guardrail_drops_total` and returned in
  `violations`.
- **Judge pipeline**: every Nth request scored 0–1 (rules judge now; LLM judge
  swaps in via the same score/threshold contract). Below `0.6` → `ALERT` log +
  `skill_judge_alerts_total` + optional `ALERT_WEBHOOK` POST.
- **Audit**: `logs/audit.jsonl` (gitignored) — `{ts, cid, text_sha256, status,
  model, n_hits, judge, cost}`. Raw text only with `SKILL_AUDIT_TEXT=1`.
- **Correlation ID**: `x-correlation-id` in → echoed in header + body + audit.

## Monitoring

```powershell
docker compose -f deploy/docker-compose.yml up   # Prometheus :9090, Grafana :3000 (admin/skilldetect)
```

Dashboard panels: req/s, latency p50/p95, total cost USD, judge avg, alerts,
circuit state, guardrail drops, tokens in/out. Alerting rule suggestion:
`skill_judge_alerts_total` rate > 0 or `skill_circuit_state == 1` → page.

## Hugging Face credential (you do this — I can't)

I cannot create accounts, receive verification email, or accept the Gemma
license as you. 2 minutes:

1. https://huggingface.co/join → sign up, verify email.
2. Open https://huggingface.co/google/gemma-3-4b-it → **Agree and access**.
3. Settings → Access Tokens → New (fine-grained, read-only; add write only if
   you want me to push the adapter).
4. In the venv: `huggingface-cli login`, paste the token (or set `HF_TOKEN`).
5. Tell me `token ready` → I push `adapters/smollm2-proof` and pull Gemma.
