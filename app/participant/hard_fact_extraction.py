from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from app.models.schemas import HardFilters

_client: anthropic.AsyncAnthropic | None = None

# Cache keyed by (query, history_length) — shared with soft_fact_extraction.
# Cleared on each new extraction to keep only the latest request's data.
_extraction_cache: dict[tuple[str, int], tuple[HardFilters, dict[str, Any]]] = {}

_SYSTEM_PROMPT = """\
You are a Swiss real-estate query parser. Extract BOTH hard constraints (must-have filters) \
and soft preferences (nice-to-have ranking signals) from natural-language queries.

Output ONLY a valid JSON object with exactly two keys: "hard" and "soft".

"hard" schema (use null for any field not explicitly stated):
{
  "city": ["Swiss city names, e.g. Zürich, Genève, Basel, Bern, Luzern, Winterthur, Zug"] | null,
  "postal_code": ["Swiss 4-digit postal codes as strings"] | null,
  "canton": "two-letter Swiss canton abbreviation e.g. ZH, GE, BS, BE, LU, ZG, VD" | null,
  "min_price": integer CHF/month or null,
  "max_price": integer CHF/month or null,
  "min_rooms": float or null,
  "max_rooms": float or null,
  "features": ["only from: balcony, elevator, parking, garage, fireplace, child_friendly, pets_allowed, new_build, wheelchair_accessible, private_laundry, minergie_certified"] | null,
  "offer_type": "RENT" | "SALE" | null,
  "object_category": ["Wohnung", "Haus", "Büro", "Gewerbe", "Studio"] | null
}

"soft" schema (use null for signals not mentioned):
{
  "brightness": 0.0-1.0 or null,
  "modernity": 0.0-1.0 or null,
  "quietness": 0.0-1.0 or null,
  "spaciousness": 0.0-1.0 or null,
  "views": 0.0-1.0 or null,
  "family_friendly": 0.0-1.0 or null,
  "commute_priority": 0.0-1.0 or null,
  "value_priority": 0.0-1.0 or null,
  "nature_proximity": 0.0-1.0 or null,
  "preferred_features": ["feature names desired but NOT hard requirements"],
  "negative_signals": ["terms to avoid, only very explicit exclusions"],
  "keywords": ["all meaningful query terms for text search, include German synonyms"],
  "inferred_price_sensitivity": "high" | "medium" | "low" | null
}

Critical Swiss rules:
- "günstig"/"nicht zu teuer"/"affordable"/"cheap"/"budget" → value_priority=0.8, inferred_price_sensitivity="high", NEVER set max_price from these words
- "under X CHF"/"unter X CHF"/"max X CHF"/"bis X CHF" → max_price=X (hard)
- "ideally with X"/"am liebsten"/"wäre schön"/"wenn möglich" → preferred_features, NOT hard features
- "with balcony"/"mit Balkon" (stated as requirement) → features=["balcony"]
- "3.5 Zimmer" → min_rooms=3.5, max_rooms=3.5 exactly; "3-4 Zimmer" → min_rooms=3.0, max_rooms=4.0
- "hell"/"lichtdurchflutet"/"sonnig"/"viel Licht"/"bright"/"light-filled" → brightness=0.9
- "modern"/"zeitgemäss"/"renoviert"/"stylish"/"zeitgemäß" → modernity=0.8
- "ruhig"/"quiet"/"leise"/"peaceful" → quietness=0.9
- "familienfreundlich"/"family-friendly"/"kinderfreundlich"/"Kinder"/"Familie" → family_friendly=0.9
- "nah am ÖV"/"gute Verkehrsanbindung"/"close to transport"/"commute"/"U-Bahn"/"S-Bahn" → commute_priority=0.9
- "Studio"/"1-Zimmer" → min_rooms=1.0, max_rooms=1.5, object_category=["Wohnung"]
- City aliases: Zurich/Zuerich → "Zürich"; Geneva/Genf → "Genève"; Berne → "Bern"; Lucerne → "Luzern"; Lausanne stays "Lausanne"
- keywords should include both original terms AND German synonyms: "bright"→["hell","sonnig","lichtdurchflutet","bright"]
- Never invent constraints. When in doubt, omit (null) rather than guess.
- Mention of neighborhood (Kreis 4, Kreis 5, Gundeldingen, Oerlikon) → add to keywords, do NOT set city unless you also know the city
- RENT is the default offer_type for Swiss listings if not specified; only set SALE if explicitly stated

Multi-turn conversation rules:
- When prior conversation context is provided, inherit all filters from the previous turn unless the current query explicitly changes or removes them
- Resolve relative references: "cheaper" → lower max_price; "larger" → raise min_rooms; "same city but quieter" → keep city, add quietness
- "show me more" / "andere zeigen" → keep all previous filters unchanged
- If the user removes a constraint ("no price limit"), set that field to null
"""


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        _client = anthropic.AsyncAnthropic(api_key=api_key)
    return _client


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = ["Prior conversation context (inherit and update these filters):\n"]
    for i, turn in enumerate(history, 1):
        hard = turn.get("hard_filters") or {}
        hard_parts = [
            f"{k}={v}" for k, v in hard.items()
            if k not in ("limit", "offset", "sort_by") and v is not None
        ]
        hard_str = ", ".join(hard_parts) if hard_parts else "no hard filters"
        lines.append(f'[{i}] User: "{turn["query"]}"')
        lines.append(f"     Resolved to: {hard_str}")
    lines.append("\n")
    return "\n".join(lines)


async def _call_claude(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> tuple[HardFilters, dict[str, Any]]:
    try:
        client = _get_client()
        history_text = _format_history(history or [])
        user_content = (
            f"{history_text}"
            f"Query: {query}\n\n"
            "Output JSON only:"
        )
        message = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except Exception:
        return HardFilters(), {"raw_query": query, "keywords": [], "preferred_features": [], "negative_signals": []}

    hard = _parse_hard(data.get("hard") or {}, query)
    soft = _parse_soft(data.get("soft") or {}, query)
    return hard, soft


def _parse_hard(h: dict[str, Any], query: str) -> HardFilters:
    try:
        cleaned: dict[str, Any] = {}
        for k, v in h.items():
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            cleaned[k] = v
        return HardFilters.model_validate(cleaned)
    except Exception:
        return HardFilters()


def _parse_soft(s: dict[str, Any], query: str) -> dict[str, Any]:
    result: dict[str, Any] = {"raw_query": query}
    float_fields = ["brightness", "modernity", "quietness", "spaciousness", "views",
                    "family_friendly", "commute_priority", "value_priority", "nature_proximity"]
    for f in float_fields:
        val = s.get(f)
        if val is not None:
            try:
                result[f] = float(val)
            except (TypeError, ValueError):
                pass
    result["preferred_features"] = [str(x) for x in (s.get("preferred_features") or [])]
    result["negative_signals"] = [str(x) for x in (s.get("negative_signals") or [])]
    result["keywords"] = [str(x) for x in (s.get("keywords") or [])]
    if s.get("inferred_price_sensitivity") in ("high", "medium", "low"):
        result["inferred_price_sensitivity"] = s["inferred_price_sensitivity"]
    return result


async def extract_hard_facts(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> HardFilters:
    key = (query, len(history or []))
    if key not in _extraction_cache:
        hard, soft = await _call_claude(query, history)
        _extraction_cache.clear()
        _extraction_cache[key] = (hard, soft)
    return _extraction_cache[key][0]
