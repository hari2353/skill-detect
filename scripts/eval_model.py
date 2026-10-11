"""Score a local base (+adapter) on heldout via the same gateway protocol.

Real-model end-to-end proof: base/adapter generate, SkillExtractor parses,
run_heldout reports F1. Wires local HF inference through CompletionClient,
so swapping fake -> real changes one constructor, not the pipeline (DIP).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts (eval_f1)

import eval_f1
from skill_project.gateway import HFClient, SkillExtractor
from skill_project.taxonomy import CompositeTaxonomy, CustomSkillRegistry, load_esco_snapshot


def build_taxonomy(esco_path: str, heldout_path: str) -> CompositeTaxonomy:
    base = load_esco_snapshot(esco_path)
    custom = CustomSkillRegistry()
    for _text, gold in eval_f1.read_heldout(Path(heldout_path)):
        for sid in gold:
            if base.lookup(sid) is None and custom.lookup(sid) is None:
                custom.register(sid, sid.split(":")[-1].replace("-", " "))
    layers: list = [base, custom]
    return CompositeTaxonomy(layers)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--heldout", default="data/heldout.jsonl")
    ap.add_argument("--esco", default="data/esco_skills.jsonl")
    args = ap.parse_args(argv)
    taxonomy = build_taxonomy(args.esco, args.heldout)
    extractor = SkillExtractor(HFClient(args.model, args.adapter), args.model, taxonomy)
    print(json.dumps(eval_f1.run_heldout(args.heldout, extractor)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
