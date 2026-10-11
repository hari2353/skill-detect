"""LiteLLM gateway: client protocol, extractor, fallback router.

Callers depend on protocols (DIP). litellm is imported lazily so unit tests
and `pip install -e .[dev]` never need it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .taxonomy import CompositeTaxonomy, Skill


@dataclass(frozen=True)
class SkillHit:
    skill_id: str
    label: str
    namespace: str
    span: str = ""
    confidence: float = 1.0


class CompletionClient(Protocol):
    def complete(self, model: str, prompt: str) -> str: ...


class FakeClient:
    """Deterministic test double. Maps prompt substring -> canned JSON response."""

    def __init__(self, response: str = "[]") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, model: str, prompt: str) -> str:
        self.calls.append((model, prompt))
        return self.response


class LiteLLMClient:
    """Production client. Requires `pip install -e .[gateway]`."""

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    def complete(self, model: str, prompt: str) -> str:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError("litellm not installed: pip install -e '.[gateway]'") from exc
        resp = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}], **self._kwargs
        )
        return resp.choices[0].message.content or "[]"


EXTRACTION_PROMPT = (
    "Extract skills from the text below. Reply ONLY with a JSON array of "
    '{{"skill_id": ..., "span": ...}} using these known IDs when they match: {known_ids}. '
    'Use "custom:<slug>" for anything else. Text: {text}'
)


class SkillExtractor:
    """Single-model extractor: prompt -> JSON -> validated SkillHits."""

    def __init__(self, client: CompletionClient, model: str, taxonomy: CompositeTaxonomy) -> None:
        self._client = client
        self._model = model
        self._taxonomy = taxonomy

    def extract(self, text: str) -> list[SkillHit]:
        known = [s.id for s in list(self._taxonomy.all_skills())[:50]]
        prompt = EXTRACTION_PROMPT.format(known_ids=", ".join(known), text=text[:2000])
        raw = self._client.complete(self._model, prompt)
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []
        hits: list[SkillHit] = []
        for it in items:
            if not isinstance(it, dict) or "skill_id" not in it:
                continue
            skill = self._taxonomy.lookup(it["skill_id"])
            if skill is None:  # custom slug invented by model: accept, namespace it
                skill = Skill(id=it["skill_id"], label=it["skill_id"].split(":")[-1].replace("-", " "),
                              namespace="custom")
            hits.append(SkillHit(skill_id=skill.id, label=skill.label,
                                 namespace=skill.namespace, span=str(it.get("span", ""))))
        return hits


class SkillRouter:
    """Try custom-adapter endpoint -> base model -> fallback. Order is data, not code."""

    def __init__(self, client: CompletionClient, taxonomy: CompositeTaxonomy,
                 chain: list[str]) -> None:
        if not chain:
            raise ValueError("chain needs at least one model")
        self._client = client
        self._taxonomy = taxonomy
        self._chain = chain
        self.used_model: str | None = None

    def extract(self, text: str) -> list[SkillHit]:
        last: Exception | None = None
        for model in self._chain:
            try:
                hits = SkillExtractor(self._client, model, self._taxonomy).extract(text)
                self.used_model = model
                return hits
            except Exception as exc:  # endpoint down -> next in chain
                last = exc
        raise RuntimeError(f"all models in chain failed: {self._chain}") from last
