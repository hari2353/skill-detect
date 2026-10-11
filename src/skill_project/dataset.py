"""Instruction-tuning dataset builder + validation (SRP: records only)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .taxonomy import CompositeTaxonomy

SYSTEM = ("You are a skill extractor. Reply ONLY with a JSON array of "
          '{"skill_id": ..., "span": ...}.')


@dataclass(frozen=True)
class TrainRecord:
    text: str
    skills: tuple[tuple[str, str], ...]  # (skill_id, span)

    def to_chat(self) -> dict:
        user = f"Extract skills. Known IDs: {', '.join(s for s, _ in self.skills)}. Text: {self.text}"
        assistant = json.dumps([{"skill_id": s, "span": sp} for s, sp in self.skills])
        return {"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]}


def validate_record(rec: TrainRecord, taxonomy: CompositeTaxonomy) -> list[str]:
    errors: list[str] = []
    if not rec.text.strip():
        errors.append("empty text")
    if not rec.skills:
        errors.append("no skills")
    for sid, span in rec.skills:
        if taxonomy.lookup(sid) is None:
            errors.append(f"unknown skill id: {sid}")
        if span and span not in rec.text:
            errors.append(f"span not in text: {span!r}")
    return errors


def write_jsonl(records: Iterable[TrainRecord], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_chat(), ensure_ascii=False) + "\n")
            n += 1
    return n
