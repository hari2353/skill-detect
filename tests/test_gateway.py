import json

from skill_project.gateway import FakeClient, SkillExtractor, SkillRouter
from skill_project.taxonomy import CompositeTaxonomy, CustomSkillRegistry, InMemoryTaxonomy, Skill


def _tax() -> CompositeTaxonomy:
    base = InMemoryTaxonomy([Skill(id="esco:1", label="Python programming", namespace="esco")])
    base.namespace = "esco"
    custom = CustomSkillRegistry([Skill(id="custom:prompt-engineering", label="prompt engineering",
                                        namespace="custom")])
    return CompositeTaxonomy([base, custom])


def test_extractor_maps_known_ids_and_spans():
    payload = json.dumps([{"skill_id": "esco:1", "span": "Python"}])
    hits = SkillExtractor(FakeClient(payload), "m", _tax()).extract("I know Python well")
    assert len(hits) == 1 and hits[0].skill_id == "esco:1" and hits[0].span == "Python"


def test_extractor_returns_empty_on_garbage():
    assert SkillExtractor(FakeClient("not json"), "m", _tax()).extract("x") == []


def test_router_tries_chain_in_order_and_records_model():
    class Flaky(FakeClient):
        def __init__(self):
            super().__init__("[]")
            self.n = 0

        def complete(self, model, prompt):
            self.calls.append((model, prompt))
            self.n += 1
            if self.n == 1:
                raise ConnectionError("adapter down")
            return "[]"

    client = Flaky()
    router = SkillRouter(client, _tax(), ["skill-detect-qlora", "skill-detect-fallback"])
    assert router.extract("hello") == []
    assert router.used_model == "skill-detect-fallback"
    assert [m for m, _ in client.calls] == ["skill-detect-qlora", "skill-detect-fallback"]


def test_router_rejects_empty_chain():
    try:
        SkillRouter(FakeClient(), _tax(), [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
