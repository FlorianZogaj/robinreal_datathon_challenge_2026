from __future__ import annotations

from typing import Any

from app.participant.hard_fact_extraction import _call_claude, _extraction_cache


def extract_soft_facts(query: str) -> dict[str, Any]:
    if query in _extraction_cache:
        return _extraction_cache[query][1]
    # Fallback: called before extract_hard_facts (shouldn't happen in normal flow)
    _, soft = _call_claude(query)
    return soft
