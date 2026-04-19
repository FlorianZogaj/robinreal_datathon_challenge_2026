from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Any

import anthropic

from app.core.image_search import image_similarity_scores
from app.core.s3 import presign_image_urls
from app.models.schemas import ListingData, RankedListingResult

_rerank_client: anthropic.AsyncAnthropic | None = None

_RERANK_SYSTEM = """\
You are a real-estate relevance judge for Swiss listings. Given a user query and candidate listings, \
score each listing's soft relevance on a scale of 0.0 to 1.0 and write a one-sentence reason.

Assume all listings already satisfy the hard constraints (rooms, price, city). Focus ONLY on soft signals: \
brightness, modernity, quietness, views, family-friendliness, commute, lifestyle fit, and overall appeal.

Output ONLY a JSON array, sorted by score descending:
[{"listing_id": "...", "score": 0.0-1.0, "reason": "one sentence"}, ...]
"""


async def rank_listings(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[RankedListingResult]:
    if not candidates:
        return []

    query_terms = _query_terms(soft_facts)
    docs = [_doc_tokens(c) for c in candidates]
    avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
    idf = _compute_idf(query_terms, docs)

    candidate_stats = _compute_candidate_stats(candidates)

    # Image similarity scores (no-op if embeddings file not yet available)
    raw_query = soft_facts.get("raw_query", "")
    img_scores: dict[str, float] = {}
    if raw_query:
        listing_ids = [str(c["listing_id"]) for c in candidates]
        img_scores = image_similarity_scores(raw_query, listing_ids)
    use_images = bool(img_scores)

    scored: list[tuple[float, dict[str, Any]]] = []
    breakdowns: dict[str, dict[str, Any]] = {}
    for c, doc in zip(candidates, docs):
        t = _bm25(query_terms, doc, avgdl, idf)
        f = _feature_score(c, soft_facts)
        g = _geo_score(c, soft_facts)
        a, a_detail = _soft_attr_score(c, soft_facts, candidate_stats)
        lm = _landmark_score(c, soft_facts)
        i = img_scores.get(str(c["listing_id"])) if use_images else None
        if lm is not None and i is not None:
            total = 0.25 * lm + 0.15 * i + 0.22 * t + 0.15 * f + 0.12 * g + 0.11 * a
            bd: dict[str, Any] = {
                "landmark": round(lm, 4), "landmark_w": 0.25,
                "image": round(i, 4), "image_w": 0.15,
                "text": round(t, 4), "text_w": 0.22,
                "feature": round(f, 4), "feature_w": 0.15,
                "geo": round(g, 4), "geo_w": 0.12,
                "soft_attr": round(a, 4), "soft_attr_w": 0.11,
                "soft_attr_detail": a_detail,
            }
        elif lm is not None:
            total = 0.30 * lm + 0.245 * t + 0.175 * f + 0.140 * g + 0.140 * a
            bd = {
                "landmark": round(lm, 4), "landmark_w": 0.30,
                "text": round(t, 4), "text_w": 0.245,
                "feature": round(f, 4), "feature_w": 0.175,
                "geo": round(g, 4), "geo_w": 0.14,
                "soft_attr": round(a, 4), "soft_attr_w": 0.14,
                "soft_attr_detail": a_detail,
            }
        elif i is not None:
            total = 0.20 * i + 0.28 * t + 0.20 * f + 0.16 * g + 0.16 * a
            bd = {
                "image": round(i, 4), "image_w": 0.20,
                "text": round(t, 4), "text_w": 0.28,
                "feature": round(f, 4), "feature_w": 0.20,
                "geo": round(g, 4), "geo_w": 0.16,
                "soft_attr": round(a, 4), "soft_attr_w": 0.16,
                "soft_attr_detail": a_detail,
            }
        else:
            total = 0.35 * t + 0.25 * f + 0.20 * g + 0.20 * a
            bd = {
                "text": round(t, 4), "text_w": 0.35,
                "feature": round(f, 4), "feature_w": 0.25,
                "geo": round(g, 4), "geo_w": 0.20,
                "soft_attr": round(a, 4), "soft_attr_w": 0.20,
                "soft_attr_detail": a_detail,
            }
        scored.append((total, c))
        breakdowns[str(c["listing_id"])] = bd

    scored.sort(key=lambda x: x[0], reverse=True)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    top_n = scored[:30]
    if api_key and len(top_n) >= 3:
        top_candidates = [c for _, c in top_n]
        rest = scored[30:]
        reranked = await _llm_rerank(top_candidates, soft_facts, api_key, breakdowns,
                                     candidate_stats, img_scores)
        return reranked + [
            _to_result(c, score=round(s, 4), reason=_formula_reason(c, soft_facts, s),
                       breakdown=breakdowns.get(str(c["listing_id"])))
            for s, c in rest
        ]

    return [
        _to_result(c, score=round(s, 4), reason=_formula_reason(c, soft_facts, s),
                   breakdown=breakdowns.get(str(c["listing_id"])))
        for s, c in scored
    ]


# ── Text scoring (BM25) ──────────────────────────────────────────────────────

def _query_terms(soft_facts: dict[str, Any]) -> list[str]:
    raw = soft_facts.get("raw_query", "")
    keywords = soft_facts.get("keywords") or []
    all_terms = _tokenize(raw) + [t.lower() for t in keywords]
    seen: set[str] = set()
    result = []
    for t in all_terms:
        if t not in seen and len(t) > 2:
            seen.add(t)
            result.append(t)
    return result


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def _doc_tokens(c: dict[str, Any]) -> list[str]:
    parts = [c.get("title") or "", c.get("description") or ""]
    return _tokenize(" ".join(parts))


def _compute_idf(query_terms: list[str], docs: list[list[str]]) -> dict[str, float]:
    N = max(len(docs), 1)
    idf: dict[str, float] = {}
    for term in query_terms:
        df = sum(1 for doc in docs if term in doc)
        idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
    return idf


def _bm25(query_terms: list[str], doc: list[str], avgdl: float,
          idf: dict[str, float], k1: float = 1.5, b: float = 0.75) -> float:
    if not query_terms:
        return 0.0
    dl = len(doc)
    tf_map = Counter(doc)
    score = 0.0
    for term in query_terms:
        f = tf_map.get(term, 0)
        if f == 0:
            continue
        tf_norm = f * (k1 + 1) / (f + k1 * (1 - b + b * dl / max(avgdl, 1)))
        score += idf.get(term, 0.5) * tf_norm
    return min(score / max(len(query_terms), 1), 1.0)


# ── Feature score ────────────────────────────────────────────────────────────

_FEATURE_COL_MAP = {
    "balcony": "feature_balcony",
    "elevator": "feature_elevator",
    "parking": "feature_parking",
    "garage": "feature_garage",
    "fireplace": "feature_fireplace",
    "child_friendly": "feature_child_friendly",
    "pets_allowed": "feature_pets_allowed",
    "new_build": "feature_new_build",
    "wheelchair_accessible": "feature_wheelchair_accessible",
    "private_laundry": "feature_private_laundry",
    "minergie_certified": "feature_minergie_certified",
}


def _candidate_features(c: dict[str, Any]) -> set[str]:
    feats: set[str] = set()
    for feat, col in _FEATURE_COL_MAP.items():
        if c.get(col) == 1:
            feats.add(feat)
    for f in (c.get("features") or []):
        feats.add(str(f).lower())
    return feats


def _feature_score(c: dict[str, Any], soft_facts: dict[str, Any]) -> float:
    wanted = set(soft_facts.get("preferred_features") or [])
    if not wanted:
        return 0.5
    have = _candidate_features(c)
    return len(wanted & have) / len(wanted)


# ── Geo/distance score ───────────────────────────────────────────────────────

def _geo_score(c: dict[str, Any], soft_facts: dict[str, Any]) -> float:
    commute = soft_facts.get("commute_priority") or 0.0
    family = soft_facts.get("family_friendly") or 0.0
    parts: list[float] = []

    dist_pt = _float(c.get("distance_public_transport"))
    if dist_pt is not None:
        parts.append(max(0.0, 1.0 - dist_pt / 1000.0) * max(commute, 0.3))

    if family > 0.1:
        for col in ("distance_kindergarten", "distance_school_1"):
            d = _float(c.get(col))
            if d is not None:
                parts.append(max(0.0, 1.0 - d / 800.0) * family)

    dist_shop = _float(c.get("distance_shop"))
    if dist_shop is not None:
        parts.append(max(0.0, 1.0 - dist_shop / 500.0) * 0.3)

    return min(sum(parts) / len(parts), 1.0) if parts else 0.5


# ── Landmark proximity score ─────────────────────────────────────────────────

def _dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _landmark_score(c: dict[str, Any], soft_facts: dict[str, Any]) -> float | None:
    """Returns 0.0–1.0 proximity score if soft_facts has landmark coords, else None."""
    lat = soft_facts.get("landmark_lat")
    lon = soft_facts.get("landmark_lon")
    c_lat = _float(c.get("latitude"))
    c_lon = _float(c.get("longitude"))
    if lat is None or lon is None or c_lat is None or c_lon is None:
        return None
    dist_km = _dist_km(lat, lon, c_lat, c_lon)
    return max(0.0, 1.0 - dist_km / 10.0)  # linear decay to 0 at 10 km


# ── Helpers ──────────────────────────────────────────────────────────────────

def _float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Candidate statistics ─────────────────────────────────────────────────────

def _compute_candidate_stats(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    vals = []
    for c in candidates:
        price = _float(c.get("price"))
        area = _float(c.get("area"))
        if price and area and area > 0:
            vals.append(price / area)
    if vals:
        vals.sort()
        median = vals[len(vals) // 2]
    else:
        median = 25.0  # fallback: ~CHF 25/m² is a typical Swiss rental benchmark
    return {"median_price_per_m2": median}


# ── Soft attribute score ─────────────────────────────────────────────────────

def _soft_attr_score(c: dict[str, Any], soft_facts: dict[str, Any],
                     candidate_stats: dict[str, Any] | None = None) -> tuple[float, dict[str, float]]:
    """Returns (aggregate 0–1, sub-scores dict keyed by signal name)."""
    score = 0.0
    total_weight = 0.0
    detail: dict[str, float] = {}

    brightness = soft_facts.get("brightness") or 0.0
    if brightness > 0.0:
        b = 0.0
        if c.get("feature_balcony") == 1:
            b += 0.4
        if c.get("feature_new_build") == 1:
            b += 0.3
        if c.get("feature_minergie_certified") == 1:
            b += 0.3
        score += b * brightness
        total_weight += brightness
        detail["brightness"] = round(b, 4)

    modernity = soft_facts.get("modernity") or 0.0
    if modernity > 0.0:
        m = 0.0
        if c.get("feature_new_build") == 1:
            m += 0.5
        if c.get("feature_minergie_certified") == 1:
            m += 0.3
        score += m * modernity
        total_weight += modernity
        detail["modernity"] = round(m, 4)

    family = soft_facts.get("family_friendly") or 0.0
    if family > 0.0:
        f = 0.0
        if c.get("feature_child_friendly") == 1:
            f += 0.5
        if c.get("feature_pets_allowed") == 1:
            f += 0.15
        d_kg = _float(c.get("distance_kindergarten"))
        if d_kg is not None and d_kg < 400:
            f += 0.35
        score += f * family
        total_weight += family
        detail["family"] = round(f, 4)

    value = soft_facts.get("value_priority") or 0.0
    if value > 0.0:
        price = _float(c.get("price")) or 0.0
        area = _float(c.get("area")) or 1.0
        price_per_m2 = price / area if area > 0 else 0.0
        median = (candidate_stats or {}).get("median_price_per_m2", 25.0)
        v = max(0.0, 1.0 - price_per_m2 / (median * 2))
        score += v * value
        total_weight += value
        detail["value"] = round(v, 4)

    for sig in ("quietness", "spaciousness", "views", "nature_proximity"):
        w = soft_facts.get(sig) or 0.0
        if w > 0.0:
            detail[sig] = 0.0  # no structural proxy yet — signal present but unscored

    agg = min(score / total_weight, 1.0) if total_weight > 0.0 else 0.5
    return agg, detail


# ── LLM re-ranking ───────────────────────────────────────────────────────────

async def _llm_rerank(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
    api_key: str,
    breakdowns: dict[str, dict[str, Any]],
    candidate_stats: dict[str, Any] | None = None,
    img_scores: dict[str, float] | None = None,
) -> list[RankedListingResult]:
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = anthropic.AsyncAnthropic(api_key=api_key)

    listings_text = "\n".join(
        _format_listing(c, soft_facts, candidate_stats, img_scores) for c in candidates
    )
    user_msg = (
        f"User query: {soft_facts.get('raw_query', '')}\n\n"
        f"Soft preferences: {_soft_summary(soft_facts, candidate_stats, bool(img_scores))}\n\n"
        f"Listings:\n{listings_text}\n\n"
        "Output JSON array sorted by score descending:"
    )

    try:
        message = await _rerank_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            temperature=0,
            system=[{"type": "text", "text": _RERANK_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        reranked_data: list[dict[str, Any]] = json.loads(raw)
    except Exception:
        return [_to_result(c, score=0.5, reason="Matched your preferences.") for c in candidates]

    by_id = {str(c["listing_id"]): c for c in candidates}
    results: list[RankedListingResult] = []
    seen: set[str] = set()

    for item in reranked_data:
        lid = str(item.get("listing_id", ""))
        if lid not in by_id or lid in seen:
            continue
        seen.add(lid)
        results.append(_to_result(
            by_id[lid],
            score=round(float(item.get("score", 0.5)), 4),
            reason=str(item.get("reason", "Relevant to your query.")),
            breakdown=breakdowns.get(lid),
        ))

    for c in candidates:
        lid = str(c["listing_id"])
        if lid not in seen:
            results.append(_to_result(c, score=0.3, reason="Matched hard filters.",
                                      breakdown=breakdowns.get(lid)))

    return results


def _soft_summary(soft_facts: dict[str, Any], candidate_stats: dict[str, Any] | None = None,
                  use_images: bool = False) -> str:
    parts = []
    for k in ("brightness", "modernity", "quietness", "spaciousness", "views",
              "family_friendly", "commute_priority"):
        v = soft_facts.get(k)
        if v and float(v) >= 0.5:
            parts.append(f"{k.replace('_', ' ')}={v:.1f}")
    v = soft_facts.get("value_priority")
    if v and float(v) >= 0.5:
        median = (candidate_stats or {}).get("median_price_per_m2")
        median_str = f" (pool median CHF {median:.0f}/m²; prefer below that)" if median else ""
        parts.append(f"value priority={v:.1f}{median_str}")
    if soft_facts.get("preferred_features"):
        parts.append(f"preferred features: {', '.join(soft_facts['preferred_features'])}")
    if soft_facts.get("keywords"):
        parts.append(f"search context: {', '.join(soft_facts['keywords'][:12])}")
    if soft_facts.get("landmark_name"):
        parts.append(f"near {soft_facts['landmark_name']} (soft proximity signal)")
    if use_images:
        parts.append("image similarity scoring active — listing image_similarity scores provided")
    return "; ".join(parts) if parts else "general relevance"


def _format_listing(c: dict[str, Any], soft_facts: dict[str, Any] | None = None,
                    candidate_stats: dict[str, Any] | None = None,
                    img_scores: dict[str, float] | None = None) -> str:
    feats = ", ".join(_candidate_features(c)) or "none"
    desc = (c.get("description") or "")[:200].replace("\n", " ")
    sf = soft_facts or {}
    signals: list[str] = []

    # Landmark proximity
    lm_lat = sf.get("landmark_lat")
    lm_lon = sf.get("landmark_lon")
    lm_name = sf.get("landmark_name", "landmark")
    c_lat = _float(c.get("latitude"))
    c_lon = _float(c.get("longitude"))
    if lm_lat is not None and lm_lon is not None and c_lat is not None and c_lon is not None:
        signals.append(f"{_dist_km(lm_lat, lm_lon, c_lat, c_lon):.1f}km from {lm_name}")

    # Image similarity
    if img_scores:
        img_sim = img_scores.get(str(c.get("listing_id", "")))
        if img_sim is not None:
            signals.append(f"image similarity {img_sim:.2f}")

    # Value vs median
    if sf.get("value_priority") and (candidate_stats or {}).get("median_price_per_m2"):
        price = _float(c.get("price"))
        area = _float(c.get("area"))
        median = candidate_stats["median_price_per_m2"]  # type: ignore[index]
        if price and area and area > 0:
            ppm2 = price / area
            pct = (ppm2 - median) / median * 100
            direction = "below" if pct < 0 else "above"
            signals.append(f"CHF {ppm2:.0f}/m² ({abs(pct):.0f}% {direction} median)")

    # Distance to public transport
    if sf.get("commute_priority") and sf["commute_priority"] > 0.1:
        d_pt = _float(c.get("distance_public_transport"))
        if d_pt is not None:
            signals.append(f"transport {int(d_pt)}m")

    # Distances to schools/kindergarten
    if sf.get("family_friendly") and sf["family_friendly"] > 0.1:
        d_kg = _float(c.get("distance_kindergarten"))
        d_sc = _float(c.get("distance_school_1"))
        if d_kg is not None:
            signals.append(f"kindergarten {int(d_kg)}m")
        if d_sc is not None:
            signals.append(f"school {int(d_sc)}m")

    signals_str = (" | " + ", ".join(signals)) if signals else ""
    return (
        f"[{c['listing_id']}] {c.get('title', 'N/A')} | "
        f"CHF {c.get('price', '?')}/mo | {c.get('rooms', '?')} rooms | {c.get('area', '?')}m² | "
        f"City: {c.get('city', '?')} | Features: {feats}{signals_str} | Desc: {desc}"
    )


# ── Reason generation ────────────────────────────────────────────────────────

def _formula_reason(c: dict[str, Any], soft_facts: dict[str, Any], score: float) -> str:
    parts: list[str] = []
    keywords = soft_facts.get("keywords") or []
    if keywords:
        doc = f"{c.get('title', '')} {c.get('description', '')}".lower()
        matched_kw = [kw for kw in keywords[:8] if kw.lower() in doc]
        if matched_kw:
            parts.append(f"Keywords matched: {', '.join(matched_kw[:3])}")
    wanted = set(soft_facts.get("preferred_features") or [])
    matched_feats = wanted & _candidate_features(c)
    if matched_feats:
        parts.append(f"Has: {', '.join(sorted(matched_feats))}")
    dist_pt = _float(c.get("distance_public_transport"))
    if dist_pt is not None and dist_pt < 400:
        parts.append(f"Transport {int(dist_pt)}m away")
    lm = _landmark_score(c, soft_facts)
    landmark_name = soft_facts.get("landmark_name")
    if lm is not None and landmark_name and lm >= 0.7:
        parts.append(f"Close to {landmark_name}")
    if not parts:
        parts.append("Matched hard filters and general relevance")
    return ". ".join(parts) + "."


def _to_result(c: dict[str, Any], score: float = 0.5, reason: str = "Matched hard filters.",
               breakdown: dict[str, Any] | None = None) -> RankedListingResult:
    return RankedListingResult(
        listing_id=str(c["listing_id"]),
        score=score,
        reason=reason,
        score_breakdown=breakdown or {},
        listing=_to_listing_data(c),
    )


def _to_listing_data(c: dict[str, Any]) -> ListingData:
    raw_urls = _coerce_image_urls(c.get("image_urls")) or []
    signed_urls = presign_image_urls(raw_urls) if raw_urls else []
    hero = signed_urls[0] if signed_urls else c.get("hero_image_url")

    return ListingData(
        id=str(c["listing_id"]),
        title=c["title"],
        description=c.get("description"),
        street=c.get("street"),
        city=c.get("city"),
        postal_code=c.get("postal_code"),
        canton=c.get("canton"),
        latitude=c.get("latitude"),
        longitude=c.get("longitude"),
        price_chf=c.get("price"),
        rooms=c.get("rooms"),
        living_area_sqm=_coerce_int(c.get("area")),
        available_from=c.get("available_from"),
        image_urls=signed_urls,
        hero_image_url=hero,
        original_listing_url=c.get("original_url"),
        features=c.get("features") or [],
        offer_type=c.get("offer_type"),
        object_category=c.get("object_category"),
        object_type=c.get("object_type"),
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
