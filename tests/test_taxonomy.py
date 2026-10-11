import json
import tempfile
from pathlib import Path

from skill_project.taxonomy import (
    CompositeTaxonomy, CustomSkillRegistry, InMemoryTaxonomy, Skill, load_esco_snapshot,
)


def _base() -> InMemoryTaxonomy:
    t = InMemoryTaxonomy([Skill(id="esco:1", label="Python programming", namespace="esco",
                                aliases=("python",))])
    t.namespace = "esco"
    return t


def test_custom_shadows_base_on_conflict():
    base = _base()
    custom = CustomSkillRegistry()
    comp = CompositeTaxonomy([base, custom])
    assert comp.lookup("esco:1").label == "Python programming"
    # same-id override lives in the later layer
    custom._by_id["esco:1"] = Skill(id="esco:1", label="Python (custom def)", namespace="custom")
    assert comp.lookup("esco:1").namespace == "custom"


def test_register_rejects_bare_ids():
    custom = CustomSkillRegistry()
    try:
        custom.register("prompt-engineering", "prompt engineering")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bare id")


def test_search_merges_layers():
    base = _base()
    custom = CustomSkillRegistry()
    custom.register("custom:prompt-engineering", "prompt engineering")
    comp = CompositeTaxonomy([base, custom])
    assert any(s.id == "custom:prompt-engineering" for s in comp.search("prompt"))
    assert any(s.id == "esco:1" for s in comp.search("python"))


def test_load_esco_snapshot_normalizes_ids():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "esco.jsonl"
        p.write_text(json.dumps({"id": "42", "label": "Welding", "aliases": ["arc welding"]}) + "\n")
        tax = load_esco_snapshot(p)
        assert tax.lookup("esco:42").label == "Welding"
        assert tax.search("arc")[0].id == "esco:42"
