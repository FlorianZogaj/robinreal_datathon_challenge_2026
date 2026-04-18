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
            _NOMINATIM_URL,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
        return (float(data[0]["lat"]), float(data[0]["lon"])) if data else None
    except Exception:
        return None
