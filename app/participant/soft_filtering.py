from __future__ import annotations

from typing import Any


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
        filtered = []
        for c in result:
            doc = f"{c.get('title', '')} {c.get('description', '')}".lower()
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
