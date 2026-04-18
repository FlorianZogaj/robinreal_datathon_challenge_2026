from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.config import get_settings
from app.core.s3 import presign_image_urls
from app.harness.search_service import query_from_filters, query_from_text
from app.harness.sessions import append_turn, get_history, new_session_id
from app.models.schemas import (
    HealthResponse,
    ListingsQueryRequest,
    ListingsResponse,
    ListingsSearchRequest,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/listings", response_model=ListingsResponse)
async def listings(request: ListingsQueryRequest) -> ListingsResponse:
    settings = get_settings()
    session_id = request.session_id or new_session_id()
    history = get_history(session_id)

    response = await query_from_text(
        db_path=settings.db_path,
        query=request.query,
        limit=request.limit,
        offset=request.offset,
        history=history,
    )

    append_turn(
        session_id=session_id,
        query=request.query,
        hard_filters=response.meta.get("extracted_hard_filters", {}),
        soft_facts=response.meta.get("extracted_soft_facts", {}),
        result_count=len(response.listings),
    )

    response.session_id = session_id
    return response


@router.post("/listings/search/filter", response_model=ListingsResponse)
async def listings_search(request: ListingsSearchRequest) -> ListingsResponse:
    settings = get_settings()
    return await query_from_filters(
        db_path=settings.db_path,
        hard_facts=request.hard_filters,
    )


@router.get("/images/proxy", include_in_schema=False)
def image_proxy(url: str) -> Response:
    """Proxy S3 images through the server to avoid browser CORS restrictions."""
    signed = presign_image_urls([url])
    target = signed[0] if signed else url
    try:
        resp = httpx.get(target, timeout=10, follow_redirects=True)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code)
        content_type = resp.headers.get("content-type", "image/jpeg")
        return Response(content=resp.content, media_type=content_type)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
