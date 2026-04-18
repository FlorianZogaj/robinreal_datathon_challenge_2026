from __future__ import annotations

import os

import httpx

# Runtime-mutable kill switch. Reads env on startup; can be toggled at runtime:
#   import app.participant.geocoding as geo; geo.enabled = False
enabled: bool = os.getenv("LANDMARK_GEOCODING", "1") != "0"

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "datathon-2026/1.0"
_TIMEOUT = 5.0
# Manually verified coordinates for landmarks Nominatim can't resolve precisely
# (e.g. multi-campus universities where Nominatim returns the wrong campus).
_KNOWN_COORDS: dict[str, tuple[float, float]] = {
    "eth zürich": (47.3764, 8.5481),       # ETH Zentrum main building
    "eth": (47.3764, 8.5481),
    "epfl lausanne": (46.5192, 6.5668),    # EPFL main campus
    "epfl": (46.5192, 6.5668),
    "zürich hb": (47.3782, 8.5403),        # Zürich Hauptbahnhof
    "zürich hauptbahnhof": (47.3782, 8.5403),
    "hauptbahnhof zürich": (47.3782, 8.5403),
    "hb zürich": (47.3782, 8.5403),
    "hb": (47.3782, 8.5403),
    "bahnhof bern": (46.9490, 7.4391),
    "bern hb": (46.9490, 7.4391),
    "bahnhof basel": (47.5477, 7.5898),
    "basel hb": (47.5477, 7.5898),
    "bahnhof lausanne": (46.5166, 6.6294),
    "lausanne hb": (46.5166, 6.6294),
}

_cache: dict[str, tuple[float, float] | None] = {}


def clear_cache() -> None:
    """Clear geocoding cache — call after changing search strategy."""
    _cache.clear()


def geocode_landmark(name: str) -> tuple[float, float] | None:
    """Return (lat, lon) for name, or None if not found / on error. Results cached."""
    known = _KNOWN_COORDS.get(name.lower().strip())
    if known is not None:
        return known
    if name in _cache:
        return _cache[name]
    # Try building/amenity types first (more precise than institution polygons),
    # then fall back to plain search, then cross-border (e.g. CERN).
    result = (
        _query_nominatim(name, countrycodes="ch", limit=5, prefer_types={"building", "amenity", "university"})
        or _query_nominatim(name, countrycodes="ch")
        or _query_nominatim(name, countrycodes=None)
    )
    _cache[name] = result
    return result


def _query_nominatim(
    name: str,
    countrycodes: str | None,
    limit: int = 1,
    prefer_types: set[str] | None = None,
) -> tuple[float, float] | None:
    params: dict = {"q": name, "format": "json", "limit": limit}
    if countrycodes:
        params["countrycodes"] = countrycodes
    try:
        r = httpx.get(
            _NOMINATIM_URL,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        if prefer_types and limit > 1:
            # Pick the first result whose type or class matches the preferred set
            for item in data:
                if item.get("type") in prefer_types or item.get("class") in prefer_types:
                    return float(item["lat"]), float(item["lon"])
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None
