from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_captions: dict[str, str] | None = None


def _get_captions() -> dict[str, str]:
    global _captions
    if _captions is None:
        p = Path(__file__).parent.parent.parent / "raw_data" / "image_captions.json"
        _captions = json.loads(p.read_text()) if p.exists() else {}
    return _captions


def filter_soft_facts(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates

    negative_signals = soft_facts.get("negative_signals") or []
    price_sensitivity = soft_facts.get("inferred_price_sensitivity")

    if not negative_signals and not price_sensitivity:
        return candidates  # fast path: nothing to filter

    result = candidates

    # Remove listings that explicitly contain terms the user wants to avoid.
    # Only trigger on terms longer than 4 chars to avoid false positives.
    long_negatives = [n.lower() for n in negative_signals if len(n) > 4]
    if long_negatives:
        captions = _get_captions()
        filtered = []
        for c in result:
            cap = captions.get(str(c.get("listing_id", "")), "")
            doc = f"{c.get('title', '')} {c.get('description', '')} {cap}".lower()
            if not any(neg in doc for neg in long_negatives):
                filtered.append(c)
        if filtered:
            result = filtered

    # For budget-sensitive queries keep the cheaper 80% by price.
    # This is a soft reduction, NOT a hard cap.
    if price_sensitivity == "high" and len(result) > 10:
        prices = sorted(c.get("price") or 0 for c in result)
        p80 = prices[int(len(prices) * 0.8)]
        filtered = [c for c in result if (c.get("price") or 0) <= p80]
        if filtered:
            result = filtered

    # Safety: never return empty — ranking will handle de-emphasis
    return result if result else candidates
