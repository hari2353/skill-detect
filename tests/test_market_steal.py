"""Market-steal additions (2026-11-18): grounded-span normalization,
tolerant JSON extraction, batch endpoint.

Compared products treat these as baseline: GLiNER-style extractors accept
case/whitespace/punctuation drift in spans; instructor/LiteLLM parsers
repair fenced or prose-wrapped JSON; production APIs expose batch
ingestion with per-item error isolation.
"""

from __future__ import annotations

import json

import pytest

from skill_project.dataset import TrainRecord, validate_record
from skill_project.gateway import FakeClient, SkillExtractor, parse_json_array
from skill_project.taxonomy import (
    CompositeTaxonomy,
    CustomSkillRegistry,
    InMemoryTaxonomy,
)
from skill_project.taxonomy import Skill

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from skill_project.serve import (  # noqa: E402
    CircuitBreaker,
    ServeSettings,
    build_app,
    span_grounded,
    validate_hits,
)


def _hit(sid: str, span: str = "Python"):
    from skill_project.gateway import SkillHit

    return SkillHit(skill_id=sid, label="Python", namespace="esco", span=span)


def _taxonomy() -> CompositeTaxonomy:
    base = InMemoryTaxonomy([Skill(id="esco:1", label="Python", namespace="esco")])
    return CompositeTaxonomy([base, CustomSkillRegistry()])


def _stub_app(hits=None, fail=False, **kw):
    class Stub:
        used_model = "skill-detect-qlora"

        def extract(self, text):
            if fail:
                raise ConnectionError("down")
            return hits or []

    kw.setdefault("audit_file", "off")
    settings = ServeSettings(sample_rate=1.0, judge_threshold=0.6, **kw)
    return build_app(
        lambda: Stub(), {"esco:1"}, settings, CircuitBreaker(failure_threshold=2)
    )


# ── grounded-span normalization ─────────────────────────────────────────────
class TestSpanGrounding:
    TEXT = "I know Python. Also  some   SQL  experience."

    def test_exact_still_wins(self) -> None:
        assert span_grounded("Python", self.TEXT)

    def test_case_drift_accepted(self) -> None:
        assert span_grounded("python", self.TEXT)

    def test_trailing_punctuation_accepted(self) -> None:
        assert span_grounded("Python.", self.TEXT)

    def test_whitespace_collapse_accepted(self) -> None:
        assert span_grounded("some SQL", self.TEXT)

    def test_truly_absent_span_rejected(self) -> None:
        assert not span_grounded("Kubernetes", self.TEXT)

    def test_empty_span_is_not_grounding(self) -> None:
        assert not span_grounded("", self.TEXT)

    def test_operator_punctuation_keeps_meaning(self) -> None:
        # "C++" must not normalize down to a bare "c"
        assert span_grounded("C++", "I write C++ daily")
        assert not span_grounded("C++", "I know C very well")
        assert span_grounded("C#", "services in C# and .NET")
        assert not span_grounded("C#", "services in C")

    def test_validate_hits_accepts_normalized_variants(self) -> None:
        hits = [_hit("esco:1", "Python."), _hit("esco:1", "kubernetes")]
        kept, violations = validate_hits(hits, self.TEXT, {"esco:1"})
        assert [h.span for h in kept] == ["Python."]
        assert any("ungrounded" in v for v in violations)

    def test_dataset_validation_accepts_normalized_variants(self) -> None:
        rec = TrainRecord(
            text="I know Python. Also some SQL.", skills=(("esco:1", "python"),)
        )
        assert validate_record(rec, _taxonomy()) == []


# ── tolerant JSON extraction ────────────────────────────────────────────────
class TestTolerantParsing:
    def test_plain_array(self) -> None:
        assert parse_json_array('[{"skill_id": "a"}]') == [{"skill_id": "a"}]

    def test_markdown_fenced_array(self) -> None:
        raw = '```json\n[{"skill_id": "a"}, {"skill_id": "b"}]\n```'
        assert len(parse_json_array(raw)) == 2

    def test_prose_wrapped_array(self) -> None:
        raw = 'Here are the skills:\n[{"skill_id": "a"}]\nHope this helps.'
        assert parse_json_array(raw) == [{"skill_id": "a"}]

    def test_single_object_becomes_one_item(self) -> None:
        assert parse_json_array('{"skill_id": "a"}') == [{"skill_id": "a"}]

    def test_brackets_inside_strings_do_not_confuse_scan(self) -> None:
        raw = 'text: [{"skill_id": "a", "span": "uses [brackets] and ] here"}]'
        items = parse_json_array(raw)
        assert items and "[brackets]" in items[0]["span"]

    def test_garbage_stays_empty(self) -> None:
        assert parse_json_array("no json at all") == []

    def test_numeric_arrays_skipped_for_real_payload(self) -> None:
        raw = 'Step [1]: here you go [{"skill_id": "a"}]'
        assert parse_json_array(raw) == [{"skill_id": "a"}]

    def test_extractor_uses_tolerant_parser(self) -> None:
        client = FakeClient(
            response='```json\n[{"skill_id": "esco:1", "span": "Python"}]\n```'
        )
        extractor = SkillExtractor(client, "m", _taxonomy())
        hits = extractor.extract("I know Python well enough.")
        assert [h.skill_id for h in hits] == ["esco:1"]
        assert extractor.parse_failures == 0

    def test_extractor_counts_parse_failures(self) -> None:
        extractor = SkillExtractor(
            FakeClient(response="oops not json"), "m", _taxonomy()
        )
        assert extractor.extract("I know Python well enough.") == []
        assert extractor.parse_failures == 1


# ── batch endpoint ──────────────────────────────────────────────────────────
class TestBatchEndpoint:
    def test_batch_isolates_failures(self) -> None:
        hits = [_hit("esco:1", "Python")]
        client = TestClient(_stub_app(hits))
        r = client.post(
            "/extract/batch",
            json={
                "texts": [
                    "I know Python well and use it daily for ETL.",
                    "   ",
                    "More Python work across teams and pipelines.",
                ]
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_ok"] == 2 and body["n_failed"] == 1
        statuses = [item["status"] for item in body["results"]]
        assert statuses == [200, 400, 200]
        assert all(
            "cid" not in item for item in body["results"]
        )  # correlation is batch-level
        assert body["correlation_id"]

    def test_batch_item_502_does_not_fail_batch(self) -> None:
        client = TestClient(_stub_app(fail=True))
        body = client.post(
            "/extract/batch",
            json={"texts": ["one long enough text here", "another long enough text"]},
        ).json()
        assert body["n_ok"] == 0 and body["n_failed"] == 2
        # threshold=2: both calls 502; the circuit opens for the next request
        assert [item["status"] for item in body["results"]] == [502, 502]
        assert client.get("/health").json()["circuit"] == "open"

    def test_batch_empty_list_rejected(self) -> None:
        client = TestClient(_stub_app([]))
        assert client.post("/extract/batch", json={"texts": []}).status_code == 400

    def test_batch_over_cap_refused_not_truncated(self) -> None:
        client = TestClient(_stub_app([]))
        r = client.post("/extract/batch", json={"texts": ["x"] * 21})
        assert r.status_code == 400 and "batch" in r.json()["detail"].lower()

    def test_batch_correlation_id_echoed(self) -> None:
        client = TestClient(_stub_app([_hit("esco:1", "Python")]))
        r = client.post(
            "/extract/batch",
            json={"texts": ["I know Python and use it daily"]},
            headers={"x-correlation-id": "batch-42"},
        )
        assert r.json()["correlation_id"] == "batch-42"
        assert r.headers["x-correlation-id"] == "batch-42"
