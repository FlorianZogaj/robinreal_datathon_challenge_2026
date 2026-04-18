from __future__ import annotations

import uuid
from typing import Any

_sessions: dict[str, list[dict[str, Any]]] = {}


def get_history(session_id: str) -> list[dict[str, Any]]:
    return list(_sessions.get(session_id, []))


def append_turn(
    session_id: str,
    query: str,
    hard_filters: dict[str, Any],
    soft_facts: dict[str, Any],
    result_count: int,
) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({
        "query": query,
        "hard_filters": hard_filters,
        "soft_facts": soft_facts,
        "result_count": result_count,
    })
    _sessions[session_id] = _sessions[session_id][-8:]


def new_session_id() -> str:
    return str(uuid.uuid4())
