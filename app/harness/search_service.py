from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.hard_filters import HardFilterParams, search_listings
from app.models.schemas import HardFilters, ListingsResponse
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.ranking import rank_listings
from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts

_SOFT_FLOAT_FIELDS = [
    "brightness", "modernity", "quietness", "spaciousness", "views",
    "family_friendly", "commute_priority", "value_priority", "nature_proximity",
]


def filter_hard_facts(db_path: Path, hard_facts: HardFilters) -> list[dict[str, Any]]:
    return search_listings(db_path, to_hard_filter_params(hard_facts))


def _merge_soft_with_history(
    soft_facts: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Carry forward soft signals from the last turn unless the current query set them."""
    if not history:
        return soft_facts
    prev_soft = history[-1].get("soft_facts") or {}
    merged = dict(soft_facts)
    for field in _SOFT_FLOAT_FIELDS:
        if merged.get(field) is None and field in prev_soft:
            merged[field] = prev_soft[field]
    # Merge preferred_features: union of old and new
    prev_feats = set(prev_soft.get("preferred_features") or [])
    curr_feats = set(merged.get("preferred_features") or [])
    if prev_feats and not curr_feats:
        merged["preferred_features"] = list(prev_feats)
    return merged


def query_from_text(
    *,
    db_path: Path,
    query: str,
    limit: int,
    offset: int,
    history: list[dict[str, Any]] | None = None,
) -> ListingsResponse:
    hard_facts = extract_hard_facts(query, history)
    hard_facts.limit = limit
    hard_facts.offset = offset
    soft_facts = extract_soft_facts(query, history)
    soft_facts = _merge_soft_with_history(soft_facts, history or [])
    candidates = filter_hard_facts(db_path, hard_facts)
    candidates = filter_soft_facts(candidates, soft_facts)
    return ListingsResponse(
        listings=rank_listings(candidates, soft_facts),
        meta={
            "extracted_hard_filters": hard_facts.model_dump(exclude_none=True),
            "extracted_soft_facts": {k: v for k, v in soft_facts.items() if k != "raw_query"},
            "candidates_after_hard_filter": len(candidates),
        },
    )


def query_from_filters(
    *,
    db_path: Path,
    hard_facts: HardFilters | None,
) -> ListingsResponse:
    structured_hard_facts = hard_facts or HardFilters()
    soft_facts = extract_soft_facts("")
    candidates = filter_hard_facts(db_path, structured_hard_facts)
    candidates = filter_soft_facts(candidates, soft_facts)
    return ListingsResponse(
        listings=rank_listings(candidates, soft_facts),
        meta={},
    )


def to_hard_filter_params(hard_facts: HardFilters) -> HardFilterParams:
    return HardFilterParams(
        city=hard_facts.city,
        postal_code=hard_facts.postal_code,
        canton=hard_facts.canton,
        min_price=hard_facts.min_price,
        max_price=hard_facts.max_price,
        min_rooms=hard_facts.min_rooms,
        max_rooms=hard_facts.max_rooms,
        latitude=hard_facts.latitude,
        longitude=hard_facts.longitude,
        radius_km=hard_facts.radius_km,
        features=hard_facts.features,
        offer_type=hard_facts.offer_type,
        object_category=hard_facts.object_category,
        limit=hard_facts.limit,
        offset=hard_facts.offset,
        sort_by=hard_facts.sort_by,
    )
