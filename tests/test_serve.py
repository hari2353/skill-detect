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
