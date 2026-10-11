"""LiteLLM gateway: client protocol, extractor, fallback router.

Callers depend on protocols (DIP). litellm is imported lazily so unit tests
and `pip install -e .[dev]` never need it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol

from .taxonomy import CompositeTaxonomy, Skill

logger = logging.getLogger(__name__)


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

    max_input_chars = 4000   # beyond this the text is chunked upstream, never silently cut
    known_id_limit = 200     # IDs listed in the prompt

    def __init__(self, client: CompletionClient, model: str, taxonomy: CompositeTaxonomy) -> None:
        self._client = client
        self._model = model
        self._taxonomy = taxonomy
        self.truncated_inputs: list[str] = []

    def extract(self, text: str) -> list[SkillHit]:
        known = [s.id for s in list(self._taxonomy.all_skills())[:self.known_id_limit]]
        if len(text) > self.max_input_chars:
            self.truncated_inputs.append(text[:80])
            raise ValueError(
                f"input too long ({len(text)} > {self.max_input_chars} chars); "
                "chunk it upstream — truncation would silently lose skills")
        prompt = EXTRACTION_PROMPT.format(known_ids=", ".join(known), text=text)
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
    """Try custom-adapter endpoint -> base model -> fallback. Order is data, not code.

    Only *availability* failures (connection/timeout/RuntimeError from the client)
    trigger the next model. Programming errors (ValueError, TypeError, ...) are
    re-raised immediately — a bug must never be silently masked as 'endpoint down'.
    """

    AVAILABILITY_ERRORS = (ConnectionError, TimeoutError, OSError, RuntimeError)

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
            except self.AVAILABILITY_ERRORS as exc:  # endpoint down -> next in chain
                logger.warning("model %s unavailable (%s); trying next in chain", model, exc)
                last = exc
        raise RuntimeError(f"all models in chain failed: {self._chain}") from last


class HFClient:
    """Local HuggingFace inference behind the CompletionClient protocol.

    `pipeline_factory(model_id, adapter_dir)` builds `prompt -> text`.
    Default factory lazily imports transformers/peft (needs `.[train]` extras).
    Inject a stub factory in unit tests — no download required.
    """

    def __init__(self, model_id: str, adapter_dir: str | None = None,
                 pipeline_factory=None, max_new_tokens: int = 256) -> None:
        self._model_id = model_id
        self._adapter_dir = adapter_dir
        self._factory = pipeline_factory or _default_hf_pipeline
        self._max_new_tokens = max_new_tokens
        self._pipe = None
        self.calls: list[tuple[str, str]] = []

    def complete(self, model: str, prompt: str) -> str:
        self.calls.append((model, prompt))
        if self._pipe is None:
            self._pipe = self._factory(self._model_id, self._adapter_dir, self._max_new_tokens)
        return self._pipe(prompt)


def _default_hf_pipeline(model_id: str, adapter_dir: str | None, max_new_tokens: int = 256):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError("HF deps missing: use the project .venv (pip install transformers peft accelerate)") from exc
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
    if adapter_dir:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft missing: pip install peft") from exc
        mdl = PeftModel.from_pretrained(mdl, adapter_dir)
    gen = pipeline("text-generation", model=mdl, tokenizer=tok)

    def run(prompt: str) -> str:
        out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
        return out[0]["generated_text"]

    return run
