from __future__ import annotations

from typing import Any

from app.participant.hard_fact_extraction import _call_claude, _extraction_cache


async def extract_soft_facts(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    key = (query, len(history or []))
    if key in _extraction_cache:
        return _extraction_cache[key][1]
    # Fallback: called before extract_hard_facts (shouldn't happen in normal flow)
    _, soft = await _call_claude(query, history)
    return soft
