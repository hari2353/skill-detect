from skill_project.dataset import TrainRecord, validate_record, write_jsonl
from skill_project.taxonomy import CompositeTaxonomy, CustomSkillRegistry, InMemoryTaxonomy, Skill


def _tax() -> CompositeTaxonomy:
    base = InMemoryTaxonomy([Skill(id="esco:1", label="Python programming", namespace="esco")])
    base.namespace = "esco"
    custom = CustomSkillRegistry([Skill(id="custom:prompt-engineering", label="prompt engineering",
                                        namespace="custom")])
    return CompositeTaxonomy([base, custom])


def test_validate_accepts_good_record():
    rec = TrainRecord(text="I know Python well", skills=(("esco:1", "Python"),))
    assert validate_record(rec, _tax()) == []


def test_validate_rejects_unknown_id_and_bad_span():
    rec = TrainRecord(text="I know Python", skills=(("esco:999", "Rust"),))
    errors = validate_record(rec, _tax())
    assert any("unknown skill id" in e for e in errors)
    assert any("span not in text" in e for e in errors)


def test_validate_rejects_empty():
    rec = TrainRecord(text="   ", skills=())
    errors = validate_record(rec, _tax())
    assert "empty text" in errors and "no skills" in errors


def test_chat_format_has_three_roles(tmp_path):
    rec = TrainRecord(text="prompt engineering expert", skills=(("custom:prompt-engineering", "prompt engineering"),))
    chat = rec.to_chat()
    assert [m["role"] for m in chat["messages"]] == ["system", "user", "assistant"]
    n = write_jsonl([rec], tmp_path / "t.jsonl")
    assert n == 1 and (tmp_path / "t.jsonl").exists()
