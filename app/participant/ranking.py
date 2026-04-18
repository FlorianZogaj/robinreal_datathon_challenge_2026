from __future__ import annotations

import json
from typing import Any

from app.core.s3 import presign_image_urls
from app.models.schemas import ListingData, RankedListingResult


def rank_listings(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[RankedListingResult]:
    """
    Basic ranking that scores candidates based on soft keyword matches.
    """
    keywords = soft_facts.get("keywords", [])
    ranked_results = []

    for candidate in candidates:
        score = 1.0  # Base score for surviving the hard filter
        match_reasons = []

        # Combine text fields to search for soft keywords
        text_to_search = (
            str(candidate.get("title", "")).lower() + " " + 
            str(candidate.get("description", "")).lower() + " " + 
            str(candidate.get("features", "")).lower()
        )

        # Boost score for each matched soft keyword
        for kw in keywords:
            if kw in text_to_search:
                score += 0.5
                match_reasons.append(kw)

        # Formulate a dynamic reason string for debugging/UI
        reason = "Matched hard filters."
        if match_reasons:
            reason += f" Soft match boost for: {', '.join(match_reasons)}."

        ranked_results.append(
            RankedListingResult(
                listing_id=str(candidate["listing_id"]),
                score=score,
                reason=reason,
                listing=_to_listing_data(candidate),
            )
        )

    # Sort the results from highest score to lowest
    ranked_results.sort(key=lambda x: x.score, reverse=True)
    
    return ranked_results


def _to_listing_data(candidate: dict[str, Any]) -> ListingData:
    raw_urls = _coerce_image_urls(candidate.get("image_urls")) or []
    signed_urls = presign_image_urls(raw_urls) if raw_urls else []
    hero = signed_urls[0] if signed_urls else candidate.get("hero_image_url")

    return ListingData(
        id=str(candidate["listing_id"]),
        title=candidate["title"],
        description=candidate.get("description"),
        street=candidate.get("street"),
        city=candidate.get("city"),
        postal_code=candidate.get("postal_code"),
        canton=candidate.get("canton"),
        latitude=candidate.get("latitude"),
        longitude=candidate.get("longitude"),
        price_chf=candidate.get("price"),
        rooms=candidate.get("rooms"),
        living_area_sqm=_coerce_int(candidate.get("area")),
        available_from=candidate.get("available_from"),
        image_urls=signed_urls,
        hero_image_url=hero,
        original_listing_url=candidate.get("original_url"),
        features=candidate.get("features") or [],
        offer_type=candidate.get("offer_type"),
        object_category=candidate.get("object_category"),
        object_type=candidate.get("object_type"),
    )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _coerce_image_urls(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return None
