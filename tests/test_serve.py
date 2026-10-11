"""Serving tests: breaker, guardrails, judge, routes. All offline with stubs."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")
from fastapi.testclient import TestClient

from skill_project.serve import (CircuitBreaker, ServeSettings, build_app, judge_score,
                                 validate_hits)


def _hit(sid, span="Python"):
    return SimpleNamespace(skill_id=sid, label="L", namespace="custom", span=span)


def test_breaker_opens_and_half_opens():
    t = [0.0]
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=10, clock=lambda: t[0])
    cb.call(lambda: 1)
    assert cb.state == "closed"
    for _ in range(2):
        try:
            cb.call(lambda: 1 / 0)
        except ZeroDivisionError:
            pass
    assert cb.state == "open"
    with pytest.raises(RuntimeError):
        cb.call(lambda: 1)
    t[0] += 11
    assert cb.call(lambda: 7) == 7 and cb.state == "closed"


def test_guardrails_drop_and_cap():
    hits = [_hit("esco:1", "Python"), _hit("bogus:x", "Python"),
            _hit("esco:2", "Rust"), _hit("custom:mine", "")]
    kept, violations = validate_hits(hits, "I know Python", {"esco:1", "esco:2"}, max_hits=3)
    # slice-then-validate: 4th hit never evaluated; unknown id + ungrounded span + cap note
    assert [h.skill_id for h in kept] == ["esco:1"]
    assert len(violations) == 3


def test_judge_bounds_and_threshold():
    assert judge_score(3, 0, 100) == 1.0
    assert judge_score(0, 0, 100) == 0.0
    assert judge_score(0, 0, 5) == 1.0  # trivial text, nothing expected
    assert 0.0 < judge_score(2, 1, 100) < 1.0


def _stub_app(hits=None, fail=False, **kw):
    class Stub:
        used_model = "skill-detect-qlora"

        def extract(self, text):
            if fail:
                raise ConnectionError("down")
            return hits or []

    kw.setdefault("audit_file", "off")
    settings = ServeSettings(sample_rate=1.0, judge_threshold=0.6, **kw)
    return build_app(lambda: Stub(), {"esco:1"}, settings, CircuitBreaker(failure_threshold=2))


def test_extract_ok_and_metrics():
    hits = [_hit("esco:1", "Python")]
    client = TestClient(_stub_app(hits))
    r = client.post("/extract", json={"text": "I know Python well and use it daily for ETL pipelines"})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["skill_id"] == "esco:1" and body["judge"] == 1.0
    assert body["cost_usd"] == 0.0  # local model is free
    assert "skill_requests_total" in client.get("/metrics").text
    assert client.get("/health").json()["circuit"] == "closed"


def test_extract_alerts_on_empty_and_breaker_opens():
    client = TestClient(_stub_app([], fail=False))
    body = client.post("/extract", json={"text": "a long enough text with no skills at all here"}).json()
    assert body["judge"] == 0.0
    assert "skill_judge_alerts_total" in client.get("/metrics").text

    bad = TestClient(_stub_app(fail=True))
    assert bad.post("/extract", json={"text": "hello world hello world"}).status_code == 502
    assert bad.post("/extract", json={"text": "hello world hello world"}).status_code == 502
    r = bad.post("/extract", json={"text": "hello world hello world"})
    assert r.status_code == 503  # circuit open
    assert bad.get("/health").json()["circuit"] == "open"


def test_empty_text_rejected():
    client = TestClient(_stub_app([]))
    assert client.post("/extract", json={"text": "   "}).status_code == 400


def test_correlation_id_echoed_or_generated():
    import json as _json
    client = TestClient(_stub_app([_hit("esco:1", "Python")]))
    r = client.post("/extract", json={"text": "Python engineer with ETL pipelines experience"},
                    headers={"x-correlation-id": "req-123"})
    assert r.status_code == 200
    assert r.json()["correlation_id"] == "req-123" == r.headers["x-correlation-id"]
    r2 = client.post("/extract", json={"text": "Python engineer with ETL pipelines experience"})
    assert len(r2.json()["correlation_id"]) == 12


def test_audit_trail_written_with_hash_not_text(tmp_path):
    import json as _json
    audit = tmp_path / "audit.jsonl"
    client = TestClient(_stub_app([_hit("esco:1", "Python")], audit_file=str(audit)))
    client.post("/extract", json={"text": "Python engineer with ETL pipelines experience"},
                headers={"x-correlation-id": "audit-1"})
    rows = [_json.loads(line) for line in audit.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["cid"] == "audit-1" and rows[0]["status"] == 200
    assert rows[0]["n_hits"] == 1 and "text" not in rows[0]  # hash only, privacy default
    assert len(rows[0]["text_sha256"]) == 16


def test_v1_models_lists_served_models():
    client = TestClient(_stub_app([_hit("esco:1", "Python")]))
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "skill-detect-qlora" in ids and all(m["object"] == "model" for m in body["data"])


def test_v1_chat_completions_wraps_extraction():
    import json as _json
    hits = [_hit("esco:1", "Python")]
    client = TestClient(_stub_app(hits))
    r = client.post("/v1/chat/completions",
                    json={"model": "m", "messages": [
                        {"role": "system", "content": "extract"},
                        {"role": "user", "content": "Python engineer with ETL pipelines"}]},
                    headers={"x-correlation-id": "oa-1"})
    assert r.status_code == 200
    b = r.json()
    assert b["object"] == "chat.completion" and b["id"] == "chatcmpl-oa-1"
    msg = b["choices"][0]["message"]
    assert msg["role"] == "assistant"
    payload = _json.loads(msg["content"])
    assert [h["skill_id"] for h in payload["hits"]] == ["esco:1"]
    assert b["usage"]["total_tokens"] > 0


def test_v1_chat_completions_last_user_message_wins():
    hits = [_hit("esco:1", "Python")]
    client = TestClient(_stub_app(hits))
    # extractor sees the LAST user turn's text, not the system or earlier turns
    seen = {}
    class Spy:
        used_model = "m"
        def extract(self, text):
            seen["text"] = text
            return hits
    app = build_app(lambda: Spy(), {"esco:1"}, ServeSettings(audit_file="off"), CircuitBreaker())
    r = TestClient(app).post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Python engineer with ETL pipelines"}]})
    assert r.status_code == 200 and seen["text"].startswith("Python engineer")


def test_empty_allowlist_accepts_base_and_namespace_ids():
    # No known_ids loaded -> esco:* accepted, custom/third_party accepted, bogus dropped
    kept, violations = validate_hits(
        [_hit("esco:1", "Python"), _hit("custom:x", ""),
         _hit("third_party:acme:y", ""), _hit("bogus:z", "")],
        "I know Python", set())
    assert [h.skill_id for h in kept] == ["esco:1", "custom:x", "third_party:acme:y"]
    assert any("bogus:z" in v for v in violations)


def test_loaded_allowlist_still_drops_unknown_esco():
    # With a real allowlist, an esco id not on it is a hallucination -> dropped
    kept, violations = validate_hits([_hit("esco:ghost", "Python")],
                                     "I know Python", {"esco:1"})
    assert kept == [] and any("esco:ghost" in v for v in violations)
