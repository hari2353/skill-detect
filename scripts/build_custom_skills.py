"""Generate synthetic third-party + custom skill training seeds (stdlib only)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TEMPLATES = [
    "Senior hire needs {label} for {ctx}.",
    "Built {ctx} using {label} and teamwork.",
    "Resume: 4 years of {label} in {ctx}.",
    "JD: must-have {label}; {ctx} experience preferred.",
]

CONTEXTS = ["ETL pipelines", "hospital triage", "retail checkout", "fleet routing", "KYC onboarding"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/custom_train.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    skills = [("custom:prompt-engineering", "prompt engineering"),
              ("custom:etl-orchestration", "ETL orchestration"),
              ("third_party:acme:forklift-safety", "forklift safety")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for i in range(args.n):
            sid, label = skills[i % len(skills)]
            text = rng.choice(TEMPLATES).format(label=label, ctx=rng.choice(CONTEXTS))
            fh.write(json.dumps({"text": text, "skills": [[sid, label]]}) + "\n")
    print(f"wrote {args.n} seeds -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
