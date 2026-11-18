"""Span grounding: match model-emitted spans to source text with tolerance
for case, whitespace and trailing-punctuation drift (GLiNER-style hygiene).
Normalization only widens matching — a span absent under every normalized
form is still rejected."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_SENTENCE_PUNCT = ".,;:!?\"'()[]{}"


def _normalize_needle(s: str) -> str:
    # only sentence-style punctuation is stripped: "C++"/"C#"/".NET" keep meaning
    return _WS.sub(" ", s.casefold()).strip().strip(_SENTENCE_PUNCT).strip()


def _normalize_text(s: str) -> str:
    return _WS.sub(" ", s.casefold())


def span_grounded(span: str, text: str) -> bool:
    """True when `span` occurs in `text`, exactly or after normalization."""
    if not span or not span.strip():
        return False
    if span in text:
        return True
    needle = _normalize_needle(span)
    return bool(needle) and needle in _normalize_text(text)
