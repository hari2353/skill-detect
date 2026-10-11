"""Micro-F1 scorer for skill-ID sets on heldout chat-JSONL.

`run_heldout(path, extractor)` is the real integration point: pass any
`SkillExtractor` (fake in tests, LiteLLM-backed live) and get aggregate scores.
CLI `--mode oracle|empty` self-tests the scorer with no model needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol


class ExtractorLike(Protocol):
    def extract(self, text: str): ...  # -> list with .skill_id


def f1_score(predicted: list[set[str]], gold: list[set[str]]) -> dict:
    tp = sum(len(p & g) for p, g in zip(predicted, gold))
    fp = sum(len(p - g) for p, g in zip(predicted, gold))
    fn = sum(len(g - p) for p, g in zip(predicted, gold))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"n": len(gold), "micro_p": round(precision, 4),
            "micro_r": round(recall, 4), "micro_f1": round(f1, 4)}


def read_heldout(path: Path) -> list[tuple[str, set[str]]]:
    pairs: list[tuple[str, set[str]]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            user = next(m["content"] for m in row["messages"] if m["role"] == "user")
            gold = {it["skill_id"] for it in json.loads(
                next(m["content"] for m in row["messages"] if m["role"] == "assistant"))}
            pairs.append((user, gold))
    return pairs


def run_heldout(path: str | Path, extractor: ExtractorLike) -> dict:
    pairs = read_heldout(Path(path))
    pred = [{h.skill_id for h in extractor.extract(text)} for text, _gold in pairs]
    gold = [g for _text, g in pairs]
    return f1_score(pred, gold)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default="data/heldout.jsonl")
    ap.add_argument("--mode", choices=["oracle", "empty"], default="oracle")
    args = ap.parse_args(argv)
    pairs = read_heldout(Path(args.heldout))

    class Oracle:
        def extract(self, text):
            from types import SimpleNamespace
            gold = next(g for t, g in pairs if t == text)
            return [SimpleNamespace(skill_id=s) for s in gold]

    class Empty:
        def extract(self, text):
            return []

    extractor = Oracle() if args.mode == "oracle" else Empty()
    print(json.dumps(run_heldout(args.heldout, extractor)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
