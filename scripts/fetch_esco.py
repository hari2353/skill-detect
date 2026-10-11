"""Fetch an ESCO skills snapshot to data/esco_skills.jsonl (JSONL: id/label/aliases).

Offline-first: the public ESCO API shape changes over time, so `--offline`
(or any network failure) writes a curated seed snapshot instead. Seeds are
clearly marked and keep the whole pipeline deterministic without network.
Real ESCO IDs drop in as a replacement file with zero code change.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ESCO_API = "https://ec.europa.eu/esco/api/resource/skill"

SEED_SKILLS: list[dict] = [
    {"id": "seed-python", "label": "Python programming", "aliases": ["python", "py"]},
    {"id": "seed-sql", "label": "SQL databases", "aliases": ["sql", "postgres", "mysql"]},
    {"id": "seed-etl", "label": "ETL pipelines", "aliases": ["etl", "data pipelines"]},
    {"id": "seed-docker", "label": "Container orchestration", "aliases": ["docker", "kubernetes", "k8s"]},
    {"id": "seed-ml", "label": "Machine learning", "aliases": ["ml", "scikit-learn"]},
    {"id": "seed-llm", "label": "Large language models", "aliases": ["llm", "transformers"]},
    {"id": "seed-java", "label": "Java programming", "aliases": ["java", "spring"]},
    {"id": "seed-js", "label": "JavaScript development", "aliases": ["javascript", "typescript", "react"]},
    {"id": "seed-pm", "label": "Project management", "aliases": ["agile", "scrum"]},
    {"id": "seed-welding", "label": "Welding", "aliases": ["arc welding", "mig"]},
    {"id": "seed-forklift", "label": "Forklift operation", "aliases": ["forklift", "material handling"]},
    {"id": "seed-nursing", "label": "Patient nursing care", "aliases": ["nursing", "triage"]},
    {"id": "seed-support", "label": "Customer support", "aliases": ["helpdesk", "customer service"]},
    {"id": "seed-accounting", "label": "Bookkeeping", "aliases": ["accounting", "ledger"]},
    {"id": "seed-excel", "label": "Spreadsheet analysis", "aliases": ["excel", "sheets"]},
]


def write_seeds(path: Path, limit: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (SEED_SKILLS * ((limit // len(SEED_SKILLS)) + 1))[:limit]
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def try_api(limit: int) -> list[dict] | None:
    """Best-effort ESCO fetch. Returns None on any failure (caller falls back).
    Probe first so an unsupported payload is reported instead of silently
    degrading to seeds."""
    try:
        url = f"{ESCO_API}?language=en&limit={min(limit, 100)}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"ESCO API probe failed ({exc}); see docs/GPU_PROOF.md note on seed fallback")
        return None
    items = payload if isinstance(payload, list) else payload.get("results", payload.get("_embedded", []))
    rows: list[dict] = []
    if isinstance(items, dict):
        items = list(items.values())
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        label = it.get("title") or it.get("preferredLabel") or it.get("label")
        uri = it.get("uri") or it.get("id")
        if label and uri:
            rows.append({"id": str(uri).split("/")[-1], "label": str(label), "aliases": []})
    if not rows:
        print("ESCO API returned no usable items; add a snapshot loader for the live schema")
    return rows or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/esco_skills.jsonl")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--offline", action="store_true", help="skip network, write seeds")
    args = ap.parse_args(argv)
    out = Path(args.out)
    rows = None if args.offline else try_api(args.limit)
    if rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows[:args.limit]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"ESCO API: wrote {min(len(rows), args.limit)} skills -> {out}")
    else:
        n = write_seeds(out, args.limit)
        print(f"seed snapshot (offline): wrote {n} skills -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
