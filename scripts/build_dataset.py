"""Build validated train/heldout instruction JSONL from ESCO + custom seeds.

Inputs:  data/esco_skills.jsonl (fetch_esco.py) + data/custom_train.jsonl (build_custom_skills.py)
Outputs: data/train.jsonl + data/heldout.jsonl (chat format, 80/20 split).
Every record is validated against the composite taxonomy; invalid rows are
dropped with a warning count (never silently trained on).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from skill_project.dataset import TrainRecord, validate_record, write_jsonl
from skill_project.taxonomy import CompositeTaxonomy, CustomSkillRegistry, load_esco_snapshot

TEMPLATES = [
    "Hands-on {label} experience in {ctx}.",
    "Resume: 3 years of {label} across {ctx}.",
    "Job requires {label}; background in {ctx} preferred.",
    "Shipped {ctx} work powered by {label}.",
]
CONTEXTS = ["ETL pipelines", "hospital triage", "retail checkout", "fleet routing", "KYC onboarding"]


def load_custom_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_records(esco_path: Path, custom_path: Path, esco_n: int, seed: int) -> tuple[list[TrainRecord], CompositeTaxonomy, int]:
    esco = load_esco_snapshot(esco_path)
    custom = CustomSkillRegistry()
    for row in load_custom_rows(custom_path):
        for sid, _label in row.get("skills", []):
            if custom.lookup(sid) is None:
                ns = "custom" if sid.startswith("custom:") else "third_party"
                custom.register(sid, sid.split(":")[-1].replace("-", " "))
    taxonomy = CompositeTaxonomy([esco, custom])
    rng = random.Random(seed)
    records: list[TrainRecord] = []
    dropped = 0

    for row in load_custom_rows(custom_path):
        for sid, label in row.get("skills", []):
            span = label if label in row["text"] else ""
            rec = TrainRecord(text=row["text"], skills=((sid, span),))
            if validate_record(rec, taxonomy):
                dropped += 1
            else:
                records.append(rec)

    for skill in list(esco.all_skills())[:esco_n]:
        text = rng.choice(TEMPLATES).format(label=skill.label, ctx=rng.choice(CONTEXTS))
        rec = TrainRecord(text=text, skills=((skill.id, skill.label),))
        if validate_record(rec, taxonomy):
            dropped += 1
        else:
            records.append(rec)
    return records, taxonomy, dropped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esco", default="data/esco_skills.jsonl")
    ap.add_argument("--custom", default="data/custom_train.jsonl")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--heldout", default="data/heldout.jsonl")
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--esco-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    records, _tax, dropped = build_records(Path(args.esco), Path(args.custom), args.esco_n, args.seed)
    if not records:
        print("no valid records — run fetch_esco.py + build_custom_skills.py first")
        return 1
    rng = random.Random(args.seed)
    rng.shuffle(records)
    cut = max(1, int(len(records) * args.heldout_frac))
    held, train = records[:cut], records[cut:]
    write_jsonl(train, args.train)
    write_jsonl(held, args.heldout)
    print(f"train={len(train)} heldout={len(held)} dropped={dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
