"""Taxonomy abstractions. SOLID: depend on SkillTaxonomy, extend by adding classes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(frozen=True)
class Skill:
    id: str          # e.g. "esco:123", "third_party:acme:456", "custom:prompt-engineering"
    label: str
    namespace: str   # "esco" | "third_party:<vendor>" | "custom"
    aliases: tuple[str, ...] = ()


class SkillTaxonomy(Protocol):
    namespace: str

    def lookup(self, skill_id: str) -> Skill | None: ...
    def search(self, query: str, limit: int = 10) -> list[Skill]: ...
    def all_skills(self) -> Iterable[Skill]: ...


class InMemoryTaxonomy:
    """Reusable base for file-backed / synthetic taxonomies."""

    namespace = "base"

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._by_id: dict[str, Skill] = {}
        for s in skills:
            self._by_id[s.id] = s

    def lookup(self, skill_id: str) -> Skill | None:
        return self._by_id.get(skill_id)

    def search(self, query: str, limit: int = 10) -> list[Skill]:
        q = query.lower()
        hits = [
            s for s in self._by_id.values()
            if q in s.label.lower() or any(q in a.lower() for a in s.aliases)
        ]
        return hits[:limit]

    def all_skills(self) -> Iterable[Skill]:
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)


class CustomSkillRegistry(InMemoryTaxonomy):
    """User/third-party skills. Custom entries shadow base IDs on conflict."""

    namespace = "custom"

    def register(self, skill_id: str, label: str, aliases: Iterable[str] = ()) -> Skill:
        if not skill_id.startswith(("custom:", "third_party:")):
            raise ValueError(f"custom id must start with custom:/third_party:, got {skill_id!r}")
        skill = Skill(id=skill_id, label=label, namespace=skill_id.split(":")[0]
                      if skill_id.startswith("third") else "custom",
                      aliases=tuple(aliases))
        self._by_id[skill.id] = skill
        return skill


class CompositeTaxonomy:
    """Base + third-party + custom. Later taxonomies win (custom > vendor > base)."""

    namespace = "composite"

    def __init__(self, layers: list[InMemoryTaxonomy]) -> None:
        if not layers:
            raise ValueError("need at least one taxonomy layer")
        self._layers = layers

    def lookup(self, skill_id: str) -> Skill | None:
        for layer in reversed(self._layers):
            hit = layer.lookup(skill_id)
            if hit is not None:
                return hit
        return None

    def search(self, query: str, limit: int = 10) -> list[Skill]:
        seen: dict[str, Skill] = {}
        for layer in reversed(self._layers):
            for s in layer.search(query, limit):
                seen.setdefault(s.id, s)
        return list(seen.values())[:limit]

    def all_skills(self) -> Iterable[Skill]:
        seen: dict[str, Skill] = {}
        for layer in self._layers:
            for s in layer.all_skills():
                seen[s.id] = s
        return list(seen.values())


def load_esco_snapshot(path: str | Path) -> InMemoryTaxonomy:
    """Load JSONL {id, label, aliases?} -> taxonomy. IDs normalized to esco:*."""
    skills: list[Skill] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row["id"] if str(row["id"]).startswith("esco:") else f"esco:{row['id']}"
            skills.append(Skill(id=sid, label=row["label"], namespace="esco",
                                aliases=tuple(row.get("aliases", []))))
    tax = InMemoryTaxonomy(skills)
    tax.namespace = "esco"
    return tax
