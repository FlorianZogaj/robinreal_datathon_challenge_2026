from __future__ import annotations

import logging
import re

import anthropic

from app.config import get_settings
from app.models.schemas import HardFilters

logger = logging.getLogger(__name__)

# ASCII -> canonical DB spelling (DB stores the German/French Unicode form)
_CITY_ALIASES: dict[str, str] = {
    "zurich": "Zürich",
    "zürich": "Zürich",
    "zuerich": "Zürich",
    "geneva": "Genf",
    "geneve": "Genève",
    "genf": "Genf",
    "bern": "Bern",
    "berne": "Bern",
    "basel": "Basel",
    "winterthur": "Winterthur",
    "lucerne": "Luzern",
    "luzern": "Luzern",
    "lausanne": "Lausanne",
    "lugano": "Lugano",
    "st. gallen": "St. Gallen",
    "saint gallen": "St. Gallen",
    "st gallen": "St. Gallen",
    "biel": "Biel/Bienne",
    "thun": "Thun",
    "zug": "Zug",
    "aarau": "Aarau",
    "schaffhausen": "Schaffhausen",
    "chur": "Chur",
    "frauenfeld": "Frauenfeld",
}


def _normalize_cities(cities: list[str] | None) -> list[str] | None:
    if not cities:
        return None
    normalized = [_CITY_ALIASES.get(c.lower(), c) for c in cities]
    return normalized or None

_VALID_FEATURES = [
    "balcony", "elevator", "parking", "garage", "fireplace",
    "child_friendly", "pets_allowed", "temporary", "new_build",
    "wheelchair_accessible", "private_laundry", "minergie_certified",
]

_EXTRACT_TOOL = {
    "name": "extract_hard_filters",
    "description": (
        "Extract hard (must-satisfy) search constraints from a natural-language "
        "real-estate query. Only populate fields that are explicitly or very clearly "
        "implied by the query. Leave everything else null."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": (
                    "List of Swiss city names the user explicitly mentions, "
                    "properly capitalised (e.g. ['Zurich', 'Winterthur']). "
                    "Use the standard German/French/Italian name."
                ),
            },
            "canton": {
                "type": ["string", "null"],
                "description": "Two-letter Swiss canton code (e.g. 'ZH', 'GE', 'BE') if stated.",
            },
            "postal_code": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Swiss postal codes if explicitly mentioned.",
            },
            "min_price": {
                "type": ["integer", "null"],
                "description": "Minimum price in CHF per month (rent) or total (sale).",
            },
            "max_price": {
                "type": ["integer", "null"],
                "description": "Maximum price in CHF (e.g. 'under 2800' -> 2800).",
            },
            "min_rooms": {
                "type": ["number", "null"],
                "description": "Minimum number of rooms (e.g. '3-room' -> 3.0).",
            },
            "max_rooms": {
                "type": ["number", "null"],
                "description": (
                    "Maximum number of rooms ONLY if the user explicitly sets an upper bound "
                    "(e.g. 'up to 4 rooms', 'max 3 rooms'). Do NOT set this when the user "
                    "says '3-room apartment' — that is a minimum, not an exact count."
                ),
            },
            "offer_type": {
                "type": ["string", "null"],
                "enum": ["RENT", "SALE", None],
                "description": "'RENT' for rental queries, 'SALE' for buy/purchase queries.",
            },
            "object_category": {
                "type": ["array", "null"],
                "items": {
                    "type": "string",
                    "enum": ["Wohnung", "Haus", "Studio", "Zimmer", "Möblierte Wohnung",
                             "Gewerbeobjekt", "Parkplatz"],
                },
                "description": (
                    "Property type(s) using the exact German DB values: "
                    "'Wohnung' (apartment/flat), 'Haus' (house/villa), "
                    "'Studio' (studio/bachelor), 'Zimmer' (single room), "
                    "'Möblierte Wohnung' (furnished apartment), "
                    "'Gewerbeobjekt' (commercial), 'Parkplatz' (parking spot). "
                    "Only set if the user explicitly specifies a property type."
                ),
            },
            "features": {
                "type": ["array", "null"],
                "items": {
                    "type": "string",
                    "enum": _VALID_FEATURES,
                },
                "description": (
                    "Hard-required building features explicitly demanded by the user. "
                    "Only include features from the allowed list."
                ),
            },
        },
        "required": [],
    },
}


def extract_hard_facts(query: str) -> HardFilters:
    api_key = get_settings().anthropic_api_key
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set – falling back to regex extraction")
        return _regex_fallback(query)

    try:
        return _llm_extract(query, api_key)
    except Exception:
        logger.exception("LLM hard-fact extraction failed – falling back to regex")
        return _regex_fallback(query)


def _llm_extract(query: str, api_key: str) -> HardFilters:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        tools=[_EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_hard_filters"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the hard search constraints from this Swiss real-estate query.\n\n"
                    f"Query: {query}"
                ),
            }
        ],
    )

    tool_use_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_use_block is None:
        return HardFilters()

    data: dict = tool_use_block.input  # type: ignore[union-attr]

    return HardFilters(
        city=_normalize_cities(data.get("city") or None),
        canton=data.get("canton") or None,
        postal_code=data.get("postal_code") or None,
        min_price=data.get("min_price"),
        max_price=data.get("max_price"),
        min_rooms=data.get("min_rooms"),
        max_rooms=data.get("max_rooms"),
        offer_type=data.get("offer_type"),
        object_category=data.get("object_category") or None,
        features=[f for f in (data.get("features") or []) if f in _VALID_FEATURES] or None,
    )


def _regex_fallback(query: str) -> HardFilters:
    query_lower = query.lower()
    filters = HardFilters()

    price_match = re.search(r'(?:under|max|maximum|bis)\s+(\d+)', query_lower)
    if price_match:
        filters.max_price = int(price_match.group(1))

    room_match = re.search(r'(\d+(?:\.\d+)?)\s*[-\s]?room', query_lower)
    if room_match:
        filters.min_rooms = float(room_match.group(1))

    swiss_cities = [
        "zurich", "zürich", "geneva", "genf", "genève", "basel",
        "bern", "berne", "winterthur", "lucerne", "luzern",
        "lausanne", "st. gallen", "lugano", "biel", "thun",
    ]
    for city in swiss_cities:
        if city in query_lower:
            filters.city = _normalize_cities([city])
            break

    return filters
