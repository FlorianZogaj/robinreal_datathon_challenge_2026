# Plan: Landmark Geocoding — Hard filter + Soft ranking signal

## Context
When a user query mentions a landmark:
- **Explicit distance** ("max 5km from ETH") → geocode → set `latitude/longitude/radius_km` on `HardFilters` → SQL haversine hard-cut
- **Vague proximity** ("near ETH") → geocode → put coords in `soft_facts` → `ranking.py` scores closer listings higher

All plumbing already exists (haversine SQL filter, `latitude/longitude/radius_km` on `HardFilters`, free `soft_facts` dict).

`main` added multi-turn history support — no conflict with our changes.

---

## New file: `app/participant/geocoding.py`

```python
from __future__ import annotations
import os
import httpx

# Runtime-mutable kill switch. Reads env on startup; can be toggled at runtime:
#   import app.participant.geocoding as geo; geo.enabled = False
enabled: bool = os.getenv("LANDMARK_GEOCODING", "1") != "0"

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "datathon-2026/1.0"
_TIMEOUT = 5.0
_cache: dict[str, tuple[float, float] | None] = {}


def geocode_landmark(name: str) -> tuple[float, float] | None:
    """Return (lat, lon) for name, or None if not found / on error. Results cached."""
    if name in _cache:
        return _cache[name]
    result = _query_nominatim(name, countrycodes="ch")
    if result is None:
        result = _query_nominatim(name, countrycodes=None)  # fallback e.g. CERN
    _cache[name] = result
    return result


def _query_nominatim(name: str, countrycodes: str | None) -> tuple[float, float] | None:
    params: dict = {"q": name, "format": "json", "limit": 1}
    if countrycodes:
        params["countrycodes"] = countrycodes
    try:
        r = httpx.get(
            _NOMINATIM_URL, params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT, follow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
        return (float(data[0]["lat"]), float(data[0]["lon"])) if data else None
    except Exception:
        return None
```

---

## Modified: `app/participant/hard_fact_extraction.py`

### Change 1 — import
```python
import app.participant.geocoding as _geocoding
from app.participant.geocoding import geocode_landmark
```

### Change 2 — system prompt: add to hard JSON schema (after `object_category`)
```
  "landmark": string | null,
  // Named building/station/institution only — NOT a bare city name.
  // Expand abbreviations: "HB" → "Zürich HB", "ETH" → "ETH Zürich", "EPFL" → "EPFL Lausanne".
  // null when query only mentions a city or canton.

  "radius_km": float | null
  // Set ONLY when query gives an explicit maximum distance:
  // "max 5km from ETH", "within 3km", "3 Kilometer Umkreis", "in 500m Umkreis".
  // Leave null for vague language ("near", "close to", "Nähe") — those are soft signals.
```

### Change 3 — Swiss rules: one bullet appended
```
- Landmark rule: "landmark" = specific named place only, never a bare city name.
  "radius_km" = explicit stated distance only; null for vague proximity ("near", "Nähe", "close to").
```

### Change 4 — geocoding wired into `_call_claude`, after both parse calls (lines 145–147)

Current:
```python
hard = _parse_hard(data.get("hard") or {}, query)
soft = _parse_soft(data.get("soft") or {}, query)
return hard, soft
```

Replace with:
```python
h = data.get("hard") or {}
hard = _parse_hard(h, query)
soft = _parse_soft(data.get("soft") or {}, query)

if _geocoding.enabled:
    landmark: str | None = h.get("landmark") or None
    raw_radius = h.get("radius_km")
    radius_km: float | None = float(raw_radius) if raw_radius is not None else None
    if landmark:
        coords = geocode_landmark(landmark)
        if coords is not None:
            if radius_km is not None:
                # Explicit distance → hard filter
                hard.latitude, hard.longitude = coords
                hard.radius_km = radius_km
            else:
                # Vague proximity → soft ranking signal
                soft["landmark_lat"] = coords[0]
                soft["landmark_lon"] = coords[1]
                soft["landmark_name"] = landmark

return hard, soft
```

No changes needed to `_parse_hard`, `_parse_soft`, or any history/cache logic.

---

## Modified: `app/participant/ranking.py`

### Change 1 — new `_landmark_score` helper (add after `_geo_score`)

`math` is already imported.

```python
def _landmark_score(c: dict[str, Any], soft_facts: dict[str, Any]) -> float | None:
    """Returns 0.0–1.0 proximity score if soft_facts has landmark coords, else None."""
    lat = soft_facts.get("landmark_lat")
    lon = soft_facts.get("landmark_lon")
    c_lat = _float(c.get("latitude"))
    c_lon = _float(c.get("longitude"))
    if lat is None or lon is None or c_lat is None or c_lon is None:
        return None
    R = 6371.0
    dlat = math.radians(c_lat - lat)
    dlon = math.radians(c_lon - lon)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(c_lat)) * math.sin(dlon / 2) ** 2
    dist_km = R * 2 * math.asin(math.sqrt(a))
    return max(0.0, 1.0 - dist_km / 10.0)  # linear decay to 0 at 10 km
```

### Change 2 — modify scoring loop in `rank_listings` (lines 41–48)

Current:
```python
for c, doc in zip(candidates, docs):
    t = _bm25(query_terms, doc, avgdl, idf)
    f = _feature_score(c, soft_facts)
    g = _geo_score(c, soft_facts)
    a = _soft_attr_score(c, soft_facts)
    total = 0.35 * t + 0.25 * f + 0.20 * g + 0.20 * a
    scored.append((total, c))
```

Replace with:
```python
for c, doc in zip(candidates, docs):
    t = _bm25(query_terms, doc, avgdl, idf)
    f = _feature_score(c, soft_facts)
    g = _geo_score(c, soft_facts)
    a = _soft_attr_score(c, soft_facts)
    lm = _landmark_score(c, soft_facts)
    if lm is not None:
        # Landmark proximity gets 0.30; other 4 signals share the remaining 0.70
        total = 0.30 * lm + 0.245 * t + 0.175 * f + 0.140 * g + 0.140 * a
    else:
        total = 0.35 * t + 0.25 * f + 0.20 * g + 0.20 * a
    scored.append((total, c))
```

### Change 3 — update `_formula_reason` to mention landmark proximity

Add after the transport proximity check (around line 327):
```python
lm = _landmark_score(c, soft_facts)
landmark_name = soft_facts.get("landmark_name")
if lm is not None and landmark_name and lm >= 0.7:
    parts.append(f"Close to {landmark_name}")
```

### Change 4 — update `_soft_summary` for LLM reranking context

Add to the parts list:
```python
if soft_facts.get("landmark_name"):
    parts.append(f"near {soft_facts['landmark_name']} (soft proximity signal)")
```

---

## Runtime kill switch

```python
import app.participant.geocoding as geo
geo.enabled = False   # disable
geo.enabled = True    # re-enable
```

Future UI endpoint: `POST /admin/geocoding?enabled=false` → `geo.enabled = False`.

---

## Verification

1. **Geocoding unit test (no API key):**
   ```python
   from app.participant.geocoding import geocode_landmark
   lat, lon = geocode_landmark("ETH Zürich")
   assert 47.3 < lat < 47.5 and 8.4 < lon < 8.7
   assert geocode_landmark("Nonexistent XYZ 999") is None
   ```

2. **Hard filter path** (explicit radius):
   ```bash
   curl -X POST http://localhost:8000/listings \
     -d '{"query": "flat max 2km from ETH Zürich under 3000 CHF", "limit": 10}'
   # All results within 2km of ETH; check meta.extracted_hard_filters for lat/lon
   ```

3. **Soft ranking path** (vague proximity):
   ```bash
   curl -X POST http://localhost:8000/listings \
     -d '{"query": "flat near ETH Zürich under 3000 CHF", "limit": 10}'
   # No hard cut, but ETH-adjacent listings ranked higher; check meta.extracted_soft_facts for landmark_lat/lon
   ```

4. **Kill switch:**
   ```python
   import app.participant.geocoding as geo; geo.enabled = False
   h = extract_hard_facts("flat max 2km from ETH Zürich")
   assert h.latitude is None
   ```

5. **Regression:** `pytest tests/ -q`
