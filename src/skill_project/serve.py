"""Serving: FastAPI + circuit breaker + guardrails + judge + Prometheus.

Real-model path needs the project .venv (transformers/peft); import is lazy so
unit tests and `/health` work anywhere.

Run (demo, no GPU):  `.venv/Scripts/python -m skill_project.serve`
Run (real):         `SKILL_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
                      SKILL_ADAPTER=adapters/smollm2-proof ...`
Scrape:              GET /metrics (Prometheus), dashboards in deploy/
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

PRICE_PER_MTOK = {  # USD per 1M tokens in/out; local models cost 0
    "skill-detect-qlora": (0.0, 0.0),
    "skill-detect-fallback": (0.15, 0.60),
}


def estimate_tokens(s: str) -> int:
    """Cheap token estimate (word count x 1.3). Documented approximation for
    cost accounting; replace with tokenizer counts for billing-grade numbers."""
    return max(1, int(len(s.split()) * 1.3))


class CircuitBreaker:
    """closed -> open (after N failures) -> half-open (after cooldown)."""

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0,
                 clock=time.monotonic) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.failures = 0
        self.state = "closed"
        self._opened_at = 0.0

    def call(self, fn, *args, **kwargs):
        if self.state == "open":
            if self._clock() - self._opened_at >= self.cooldown_s:
                self.state = "half-open"
            else:
                raise RuntimeError("circuit open")
        try:
            out = fn(*args, **kwargs)
        except Exception:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = "open"
                self._opened_at = self._clock()
            raise
        self.failures = 0
        self.state = "closed"
        return out


def validate_hits(hits: list, text: str, known_ids: set[str], max_hits: int = 20):
    """Guardrails: ground spans, known/custom IDs, cap count. Returns (kept, violations).

    An empty `known_ids` means "no allowlist loaded" — base-taxonomy `esco:*` ids
    are then accepted (custom:/third_party: always pass by namespace). Once a real
    allowlist is provided, an `esco:*` id not on it is treated as a hallucination
    and dropped, so the strict check is never weakened when it matters.
    """
    kept, violations = [], []
    for h in hits[:max_hits]:
        sid = h.skill_id if hasattr(h, "skill_id") else h.get("skill_id", "")
        span = h.span if hasattr(h, "span") else h.get("span", "")
        if sid.startswith("custom:") or sid.startswith("third_party:"):
            allowed = True                      # invented by design; namespace-suffixed
        elif known_ids:
            allowed = sid in known_ids          # allowlist loaded: enforce it
        else:
            allowed = sid.startswith("esco:")   # no allowlist: accept base ids only
        if not allowed:
            violations.append(f"unknown id dropped: {sid}")
            continue
        if span and span not in text:
            violations.append(f"ungrounded span dropped: {span!r}")
            continue
        kept.append(h)
    if len(hits) > max_hits:
        violations.append(f"capped {len(hits)} -> {max_hits}")
    return kept, violations


def judge_score(n_hits: int, n_violations: int, text_len: int) -> float:
    """Offline rules judge (no LLM call): 1.0 = clean extraction, 0.0 = empty
    on a substantive text or all output dropped. Swap for an LLM judge via the
    same (score, threshold) contract when keys are available."""
    if text_len < 20:
        return 1.0 if n_hits == 0 else 0.7
    if n_hits == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - 0.25 * n_violations))


@dataclass
class ServeSettings:
    model_name: str = "skill-detect-qlora"
    judge_threshold: float = 0.6
    sample_rate: float = 0.2
    max_inflight: int = 4
    max_hits: int = 20
    max_input_chars: int = 4000  # hard cap on request text; longer is refused, not truncated
    webhook: str = ""
    audit_file: str = "logs/audit.jsonl"  # "off" disables; SKILL_AUDIT_TEXT=1 logs raw text
    include_text: bool = False


def build_metrics(registry: CollectorRegistry) -> dict:
    return {
        "requests": Counter("skill_requests_total", "extract requests", ["status"], registry=registry),
        "latency": Histogram("skill_request_latency_seconds", "extract latency", registry=registry),
        "inflight": Gauge("skill_inflight_requests", "active extracts", registry=registry),
        "judge": Histogram("skill_judge_score", "sampled judge scores", registry=registry),
        "alerts": Counter("skill_judge_alerts_total", "judge below threshold", registry=registry),
        "drops": Counter("skill_guardrail_drops_total", "hits dropped by guardrails", ["reason"], registry=registry),
        "cost": Counter("skill_cost_usd_total", "estimated USD", ["model"], registry=registry),
        "tokens_in": Counter("skill_tokens_in_total", "est. input tokens", registry=registry),
        "tokens_out": Counter("skill_tokens_out_total", "est. output tokens", registry=registry),
        "circuit": Gauge("skill_circuit_state", "0 closed, 1 open", registry=registry),
    }


def build_app(make_extractor=None, known_ids: set[str] | None = None,
              settings: ServeSettings | None = None,
              breaker: CircuitBreaker | None = None) -> FastAPI:
    settings = settings or ServeSettings()
    breaker = breaker or CircuitBreaker()
    known_ids = known_ids or set()
    registry = CollectorRegistry()
    m = build_metrics(registry)
    gate = asyncio.Semaphore(settings.max_inflight)
    state = {"extractor": None, "calls": 0}

    def get_extractor():
        if state["extractor"] is None:
            state["extractor"] = make_extractor() if make_extractor else _default_extractor()
        return state["extractor"]

    app = FastAPI(title="skill-detect")

    def audit(record: dict) -> None:
        """Append-only JSONL audit. Logs text hash + length by default (privacy);
        raw text only when settings.include_text. Best-effort, never fails requests."""
        if settings.audit_file == "off":
            return
        try:
            os.makedirs(os.path.dirname(settings.audit_file) or ".", exist_ok=True)
            with open(settings.audit_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            print(f"audit write failed (best-effort): {exc}")

    @app.middleware("http")
    async def correlation_id(request: Request, call_next):
        cid = request.headers.get("x-correlation-id", uuid.uuid4().hex[:12])
        request.state.correlation_id = cid
        resp = await call_next(request)
        resp.headers["x-correlation-id"] = cid
        return resp

    @app.get("/health")
    def health():
        return {"status": "ok", "model": settings.model_name, "circuit": breaker.state}

    @app.get("/metrics")
    def metrics():
        m["circuit"].set(1 if breaker.state == "open" else 0)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    async def run_pipeline(text: str, cid: str) -> dict:
        """Core extraction: breaker, guardrails, judge, metrics, audit.

        Shared by the native /extract and the OpenAI-compatible
        /v1/chat/completions so the safety path is never duplicated.
        """
        base_audit = {"ts": datetime.now(timezone.utc).isoformat(), "cid": cid,
                      "text_len": len(text),
                      "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
        if settings.include_text:
            base_audit["text"] = text
        if not text.strip():
            audit({**base_audit, "status": 400})
            raise HTTPException(400, "empty text")
        if gate.locked():
            m["requests"].labels("busy").inc()
            audit({**base_audit, "status": 503})
            raise HTTPException(503, "server saturated, retry later")
        t0 = time.monotonic()
        async with gate:
            m["inflight"].inc()
            try:
                hits = await asyncio.to_thread(
                    lambda: breaker.call(get_extractor().extract, text))
            except RuntimeError as exc:
                m["requests"].labels("open_circuit" if "open" in str(exc) else "error").inc()
                audit({**base_audit, "status": 503, "circuit": breaker.state})
                raise HTTPException(503, str(exc))
            except Exception as exc:
                m["requests"].labels("error").inc()
                audit({**base_audit, "status": 502, "error": str(exc)[:200]})
                raise HTTPException(502, f"extraction failed: {exc}")
            finally:
                m["inflight"].dec()
        kept, violations = validate_hits(hits, text, known_ids, settings.max_hits)
        for v in violations:
            m["drops"].labels("guardrail").inc()
        used = getattr(get_extractor(), "used_model", None) or settings.model_name
        in_tok, out_tok = estimate_tokens(text), sum(estimate_tokens(h.span) for h in kept)
        pin, pout = PRICE_PER_MTOK.get(used, (0.0, 0.0))
        cost = (in_tok * pin + out_tok * pout) / 1e6
        m["tokens_in"].inc(in_tok)
        m["tokens_out"].inc(out_tok)
        m["cost"].labels(used).inc(cost)
        score = None
        state["calls"] += 1
        if settings.sample_rate >= 1.0 or (state["calls"] - 1) % max(1, int(1 / settings.sample_rate)) == 0:
            score = judge_score(len(kept), len(violations), len(text))
            m["judge"].observe(score)
            if score < settings.judge_threshold:
                m["alerts"].inc()
                print(f"ALERT judge {score:.2f} < {settings.judge_threshold} text={text[:80]!r}")
                if settings.webhook:
                    _post_webhook(settings.webhook, score, text)
        m["latency"].observe(time.monotonic() - t0)
        m["requests"].labels("ok").inc()
        audit({**base_audit, "status": 200, "model": used, "n_hits": len(kept),
               "violations": violations, "judge": score, "cost_usd": round(cost, 6)})
        return {
            "correlation_id": cid,
            "hits": [{"skill_id": h.skill_id, "label": h.label,
                      "namespace": h.namespace, "span": h.span} for h in kept],
            "used_model": used, "judge": score, "violations": violations,
            "cost_usd": round(cost, 6), "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    @app.post("/extract")
    async def extract(body: dict, request: Request):
        cid = getattr(request.state, "correlation_id", "-")
        raw = str(body.get("text", ""))
        if len(raw) > settings.max_input_chars:
            audit({"ts": datetime.now(timezone.utc).isoformat(), "cid": cid,
                   "text_len": len(raw), "status": 400})
            raise HTTPException(
                400, f"input too long ({len(raw)} > {settings.max_input_chars} chars); "
                     "chunk it upstream — truncation would silently lose skills")
        return JSONResponse(await run_pipeline(raw, cid))

    # ---- OpenAI-compatible surface (clients probe /v1/*; see configs/litellm.yaml) ----

    @app.get("/v1/models")
    def v1_models():
        # Never load the extractor just to list models (real weights are heavy
        # and may be absent); the served name is static config.
        return {"object": "list", "data": [
            {"id": settings.model_name, "object": "model", "owned_by": "skill-detect"}]}

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(body: dict, request: Request):
        """Run extraction on the last user message; return a chat.completion whose
        content is the JSON hits. Guardrails/breaker/judge all still apply."""
        cid = getattr(request.state, "correlation_id", "-")
        msgs = body.get("messages") or []
        text = ""
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = str(msg.get("content", ""))
                break
        if len(text) > settings.max_input_chars:
            raise HTTPException(
                400, f"input too long ({len(text)} > {settings.max_input_chars} chars); "
                     "chunk it upstream — truncation would silently lose skills")
        result = await run_pipeline(text, cid)
        content = json.dumps({"hits": result["hits"], "violations": result["violations"],
                              "judge": result["judge"]})
        model = body.get("model") or result["used_model"]
        return {
            "id": f"chatcmpl-{cid}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": estimate_tokens(text),
                      "completion_tokens": estimate_tokens(content),
                      "total_tokens": estimate_tokens(text) + estimate_tokens(content)},
        }

    return app


def _post_webhook(url: str, score: float, text: str) -> None:
    try:
        req = urllib.request.Request(url, data=json.dumps(
            {"alert": "judge below threshold", "score": score,
             "text": text[:200]}).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3)
    except Exception as exc:
        print(f"webhook failed (best-effort): {exc}")


def _default_extractor():
    """Demo extractor (no GPU) unless SKILL_MODEL points at real weights."""
    from skill_project.gateway import FakeClient, HFClient, SkillRouter
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"))
    import demo_extract
    taxonomy = demo_extract.build_demo_taxonomy()
    model = os.environ.get("SKILL_MODEL", "demo")
    if model == "demo":
        return SkillRouter(FakeClient(demo_extract.CANNED), taxonomy,
                           ["skill-detect-qlora", "skill-detect-fallback"])
    return SkillRouter(HFClient(model, os.environ.get("SKILL_ADAPTER") or None),
                       taxonomy, [model])


app = build_app()

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run("skill_project.serve:app",
                host=os.environ.get("SKILL_HOST", "127.0.0.1"),  # localhost by default; never 0.0.0.0 unless asked
                port=int(os.environ.get("SKILL_PORT", "8000")))
