import json

from skill_project.gateway import FakeClient, HFClient, SkillExtractor, SkillRouter
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


def _stub_factory(seen: dict):
    def factory(model_id, adapter_dir, max_new_tokens=256):
        seen["model_id"] = model_id
        seen["adapter_dir"] = adapter_dir
        seen["max_new_tokens"] = max_new_tokens
        return lambda prompt: json.dumps([{"skill_id": "esco:1", "span": "Python"}])
    return factory


def test_hfclient_uses_injected_pipeline_without_download():
    seen: dict = {}
    client = HFClient("any-model", "adapters/x", pipeline_factory=_stub_factory(seen),
                      max_new_tokens=64)
    hits = SkillExtractor(client, "any-model", _tax()).extract("I know Python")
    assert hits[0].skill_id == "esco:1"
    assert seen == {"model_id": "any-model", "adapter_dir": "adapters/x", "max_new_tokens": 64}
    assert client.calls and client.calls[0][0] == "any-model"


def test_hfclient_garbage_output_yields_no_hits():
    client = HFClient("m", pipeline_factory=lambda mid, ad, n=256: (lambda p: "not json"))
    assert SkillExtractor(client, "m", _tax()).extract("x") == []


def test_extractor_refuses_overlong_input_instead_of_truncating():
    ex = SkillExtractor(FakeClient("[]"), "m", _tax())
    try:
        ex.extract("x" * 5000)
    except ValueError as exc:
        assert "too long" in str(exc)
    else:
        raise AssertionError("expected refusal, not silent truncation")


def test_router_does_not_mask_bugs_as_outage():
    class Bug(FakeClient):
        def complete(self, model, prompt):
            raise TypeError("real bug")

    router = SkillRouter(Bug(), _tax(), ["m1", "m2"])
    try:
        router.extract("hello")
    except TypeError:
        pass
    else:
        raise AssertionError("TypeError must propagate, not trigger fallback")
