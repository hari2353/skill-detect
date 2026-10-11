"""End-to-end integration tests: real files + real pipeline, fakes only at the LLM edge.

Scenarios mirror production use: resume intake, custom-skill onboarding,
adapter outage fallback, dataset build, scoring, config validity, demo.
No network, no GPU.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_dataset
import build_custom_skills
import demo_extract
import eval_f1
import fetch_esco
from skill_project.dataset import TrainRecord, validate_record, write_jsonl
from skill_project.gateway import FakeClient, SkillExtractor, SkillRouter
from skill_project.taxonomy import CompositeTaxonomy, CustomSkillRegistry, load_esco_snapshot

RESUME = ("Hired ML engineer: Python programming for ETL pipelines, "
          "Docker deploys, and prompt engineering for support bots.")


def _snapshot(tmp_path: Path) -> Path:
    p = tmp_path / "esco.jsonl"
    for row in fetch_esco.SEED_SKILLS[:6]:
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    return p


def _taxonomy(p: Path) -> CompositeTaxonomy:
    base = load_esco_snapshot(p)
    custom = CustomSkillRegistry()
    custom.register("custom:prompt-engineering", "prompt engineering")
    return CompositeTaxonomy([base, custom])


def test_resume_to_skills_end_to_end(tmp_path):
    taxonomy = _taxonomy(_snapshot(tmp_path))
    canned = json.dumps([
        {"skill_id": "esco:seed-python", "span": "Python programming"},
        {"skill_id": "esco:seed-etl", "span": "ETL pipelines"},
        {"skill_id": "custom:prompt-engineering", "span": "prompt engineering"},
    ])
    hits = SkillExtractor(FakeClient(canned), "skill-detect-qlora", taxonomy).extract(RESUME)
    by_id = {h.skill_id: h for h in hits}
    assert by_id["esco:seed-python"].namespace == "esco"
    assert by_id["custom:prompt-engineering"].namespace == "custom"
    # every hit must validate as a training record (closed loop: extract -> train)
    rec = TrainRecord(text=RESUME, skills=tuple((h.skill_id, h.span) for h in hits))
    assert validate_record(rec, taxonomy) == []


def test_custom_skill_onboarding_flows_into_prompt(tmp_path):
    taxonomy = _taxonomy(_snapshot(tmp_path))
    client = FakeClient("[]")
    SkillExtractor(client, "m", taxonomy).extract("we need forklift safety here")
    prompt_sent = client.calls[0][1]
    assert "custom:prompt-engineering" in prompt_sent  # registered skills are known IDs


def test_adapter_outage_falls_back_and_still_extracts(tmp_path):
    taxonomy = _taxonomy(_snapshot(tmp_path))

    class Flaky(FakeClient):
        def __init__(self):
            super().__init__("[]")
            self.n = 0

        def complete(self, model, prompt):
            self.calls.append((model, prompt))
            self.n += 1
            if self.n == 1:
                raise ConnectionError("qlora endpoint down")
            return json.dumps([{"skill_id": "esco:seed-sql", "span": "SQL"}])

    router = SkillRouter(Flaky(), taxonomy, ["skill-detect-qlora", "skill-detect-fallback"])
    hits = router.extract("SQL databases expert")
    assert router.used_model == "skill-detect-fallback"
    assert hits[0].skill_id == "esco:seed-sql"


def test_dataset_pipeline_writes_valid_splits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert fetch_esco.main(["--out", "data/esco_skills.jsonl", "--limit", "30", "--offline"]) == 0
    assert build_custom_skills.main(["--out", "data/custom_train.jsonl", "--n", "30"]) == 0
    assert build_dataset.main(["--esco-n", "30", "--seed", "7"]) == 0
    taxonomy = CompositeTaxonomy([load_esco_snapshot(Path("data/esco_skills.jsonl"))])
    for f in ("data/train.jsonl", "data/heldout.jsonl"):
        lines = Path(f).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1
        for line in lines:  # every line is a valid 3-role chat with parseable gold JSON
            row = json.loads(line)
            assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
            gold = json.loads(next(m["content"] for m in row["messages"] if m["role"] == "assistant"))
            assert gold and all("skill_id" in g for g in gold)


def test_eval_f1_oracle_perfect_empty_zero(tmp_path):
    recs = [
        TrainRecord(text="Python programming guru", skills=(("esco:seed-python", "Python programming"),)),
        TrainRecord(text="SQL databases admin", skills=(("esco:seed-sql", "SQL databases"),)),
    ]
    held = tmp_path / "held.jsonl"
    write_jsonl(recs, held)
    pairs = eval_f1.read_heldout(held)

    class Oracle:
        def extract(self, text):
            from types import SimpleNamespace
            return [SimpleNamespace(skill_id=s) for _t, g in pairs if _t == text for s in g]

    class Empty:
        def extract(self, text):
            return []

    assert eval_f1.run_heldout(held, Oracle())["micro_f1"] == 1.0
    assert eval_f1.run_heldout(held, Empty())["micro_f1"] == 0.0
    partial = eval_f1.f1_score([{"a", "b"}], [{"a", "c"}])
    assert partial == {"n": 1, "micro_p": 0.5, "micro_r": 0.5, "micro_f1": 0.5}


def test_litellm_config_chain_valid():
    cfg = yaml.safe_load((ROOT / "configs" / "litellm.yaml").read_text(encoding="utf-8"))
    names = [m["model_name"] for m in cfg["model_list"]]
    assert names[0] == "skill-detect-qlora" and names[-1] == "skill-detect-fallback"
    fallbacks = cfg["router_settings"]["fallbacks"]  # LiteLLM form: [{model: [fallbacks]}]
    flat = {}
    for entry in fallbacks:
        flat.update(entry)
    assert flat["skill-detect-qlora"] == ["skill-detect-fallback"]
    # usage-based-routing balances same-model deployments, not cross-model failover
    assert cfg["router_settings"]["routing_strategy"] != "usage-based-routing"
    assert cfg["router_settings"].get("num_retries", 0) >= 1


def test_litellm_router_constructs_from_config():
    litellm = pytest.importorskip("litellm")
    cfg = yaml.safe_load((ROOT / "configs" / "litellm.yaml").read_text(encoding="utf-8"))
    rs = cfg["router_settings"]
    r = litellm.Router(model_list=cfg["model_list"], fallbacks=rs["fallbacks"],
                       routing_strategy=rs["routing_strategy"],
                       num_retries=rs["num_retries"])
    assert [m["model_name"] for m in r.get_model_list()] == [
        "skill-detect-qlora", "skill-detect-fallback"]
    assert r.fallbacks == [{"skill-detect-qlora": ["skill-detect-fallback"]}]


def test_train_script_cli_parses():
    import train_qlora
    with pytest.raises(SystemExit) as exc:
        train_qlora.main(["--help"])
    assert exc.value.code == 0


def test_esco_probe_failure_falls_back_openly(monkeypatch, tmp_path, capsys):
    import urllib.request

    def boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert fetch_esco.try_api(10) is None
    assert "probe failed" in capsys.readouterr().out


def test_lora_config_targets_gemma_on_8gb():
    cfg = yaml.safe_load((ROOT / "configs" / "lora.yaml").read_text(encoding="utf-8"))
    assert "gemma" in cfg["base_model"].lower()
    assert cfg["load_in_4bit"] is True and cfg["per_device_batch"] == 1


def test_demo_runs_offline(capsys):
    assert demo_extract.main([]) == 0
    out = capsys.readouterr().out
    assert "custom:prompt-engineering" in out and "esco:seed-python" in out
