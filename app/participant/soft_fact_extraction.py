from __future__ import annotations

import logging
from typing import Any

import anthropic

from app.config import get_settings

logger = logging.getLogger(__name__)

_EXTRACT_TOOL = {
    "name": "extract_soft_facts",
    "description": (
        "Extract soft preferences (nice-to-haves, ranking hints) from a "
        "natural-language real-estate query. These are NOT hard constraints – "
        "they influence ranking but do not filter out listings."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Short descriptive terms reflecting the user's soft preferences "
                    "(e.g. 'bright', 'modern', 'quiet', 'balcony', 'garden view', "
                    "'good transport', 'renovated', 'spacious', 'central location'). "
                    "Extract as many as clearly implied."
                ),
            },
            "preferred_locations": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Neighbourhood, district, or area names mentioned as preferences "
                    "(not as hard city filters)."
                ),
            },
            "commute_hint": {
                "type": ["string", "null"],
                "description": (
                    "A free-text commute requirement if stated, e.g. "
                    "'max 30 min to ETH Zurich by public transport'."
                ),
            },
            "availability_hint": {
                "type": ["string", "null"],
                "description": (
                    "Move-in or availability preference if stated, e.g. 'June move-in', "
                    "'available immediately'."
                ),
            },
            "price_sensitivity": {
                "type": ["string", "null"],
                "enum": ["budget", "moderate", "premium", None],
                "description": (
                    "Overall price sensitivity inferred from the query: "
                    "'budget' (cheap, affordable, student), "
                    "'moderate' (no strong signal), "
                    "'premium' (luxury, high-end, spacious)."
                ),
            },
            "summary": {
                "type": "string",
                "description": (
                    "One sentence summarising what the user is ideally looking for, "
                    "useful for generating ranking explanations."
                ),
            },
        },
        "required": ["keywords", "summary"],
    },
}


def extract_soft_facts(query: str) -> dict[str, Any]:
    if not query.strip():
        return _empty()

    api_key = get_settings().anthropic_api_key
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set – falling back to keyword extraction")
        return _keyword_fallback(query)

    try:
        return _llm_extract(query, api_key)
    except Exception:
        logger.exception("LLM soft-fact extraction failed – falling back to keywords")
        return _keyword_fallback(query)


def _llm_extract(query: str, api_key: str) -> dict[str, Any]:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        tools=[_EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_soft_facts"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the soft preferences from this Swiss real-estate query.\n\n"
                    f"Query: {query}"
                ),
            }
        ],
    )

    tool_use_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_use_block is None:
        return _keyword_fallback(query)

    data: dict = tool_use_block.input  # type: ignore[union-attr]
    return {
        "keywords": data.get("keywords") or [],
        "preferred_locations": data.get("preferred_locations") or [],
        "commute_hint": data.get("commute_hint"),
        "availability_hint": data.get("availability_hint"),
        "price_sensitivity": data.get("price_sensitivity"),
        "summary": data.get("summary", ""),
    }


def _keyword_fallback(query: str) -> dict[str, Any]:
    query_lower = query.lower()
    possible_keywords = [
        "bright", "modern", "quiet", "family-friendly", "balcony",
        "view", "central", "parking", "garden", "renovated", "spacious",
        "new", "furnished", "gym", "concierge", "terrace",
    ]
    keywords = [kw for kw in possible_keywords if kw in query_lower]
    return {
        "keywords": keywords,
        "preferred_locations": [],
        "commute_hint": None,
        "availability_hint": None,
        "price_sensitivity": None,
        "summary": query,
    }


def _empty() -> dict[str, Any]:
    return {
        "keywords": [],
        "preferred_locations": [],
        "commute_hint": None,
        "availability_hint": None,
        "price_sensitivity": None,
        "summary": "",
    }
