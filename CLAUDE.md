# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavior

**Think before coding.** State assumptions explicitly. If multiple interpretations exist, present them — don't pick silently. If something is unclear, stop and ask before implementing.

**Simplicity first.** Minimum code that solves the problem. No features beyond what was asked, no abstractions for single-use code, no error handling for impossible scenarios.

**Surgical changes.** Touch only what you must. Don't improve adjacent code or refactor things that aren't broken. Match existing style. Every changed line should trace directly to the request.

**Goal-driven.** For multi-step tasks, state a brief plan with verifiable success criteria before starting.

## Commands

```bash
uv sync --dev                                              # install deps
uv run uvicorn app.main:app --reload                       # run API (port 8000)
uv run uvicorn apps_sdk.server.main:app --reload --port 8001  # run MCP bridge
uv run pytest tests -q                                     # run all tests
uv run pytest tests/test_hard_filters.py -q               # single test file
cd apps_sdk/web && npm install && npm run build            # build widget
uv run python scripts/mcp_smoke.py --url http://localhost:8001/mcp  # MCP smoke test
```

The SQLite database is built automatically from `raw_data/*.csv` on first startup. Delete the `.db` file and restart to rebuild.

## Architecture

Pipeline orchestrated in `app/harness/search_service.py`:

```
user query → extract_hard_facts() → HardFilters
           → extract_soft_facts() → dict
           → filter_hard_facts()  → SQL against SQLite
           → filter_soft_facts()  → post-filter candidates
           → rank_listings()      → List[RankedListingResult]
```

**Participant extension points** — all stubs under `app/participant/`:
- `hard_fact_extraction.py` — NL → `HardFilters`
- `soft_fact_extraction.py` — NL → free `dict`
- `soft_filtering.py` — post-filter hard-filtered candidates
- `ranking.py` — score/sort → `List[RankedListingResult]`
- `listing_row_parser.py` — CSV row → DB tuple (feature parsing already implemented)

**Harness glue** (avoid modifying):
- `app/harness/bootstrap.py` — startup: creates SQLite from CSVs, checks schema version
- `app/harness/csv_import.py` — CSV → SQLite schema and import
- `app/core/hard_filters.py` — SQL query builder: city, canton, price, rooms, radius (haversine), feature boolean columns, offer_type, object_category

**Key types** (`app/models/schemas.py`):
- `HardFilters` — what `extract_hard_facts` must return
- `RankedListingResult` — `{listing_id, score, reason, listing: ListingData}`; what `rank_listings` must return
- `ListingsResponse.meta` — open dict for debug info, extracted filters, etc.

**API** (`app/api/routes/listings.py`):
- `POST /listings` — full NL pipeline
- `POST /listings/search/filter` — direct hard filter query (bypasses NL)

**SQLite listings columns available for scoring/ranking:** `price`, `rooms`, `area`, `latitude`, `longitude`, `distance_public_transport`, `distance_shop`, `distance_kindergarten`, `distance_school_1`, `distance_school_2`, `feature_balcony`, `feature_elevator`, `feature_parking`, `feature_garage`, `feature_fireplace`, `feature_child_friendly`, `feature_pets_allowed`, `feature_temporary`, `feature_new_build`, `feature_wheelchair_accessible`, `feature_private_laundry`, `feature_minergie_certified`, `features_json`, `images_json`, `orig_data`, `raw_row`

**S3 images** (`app/core/s3.py`): needs `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION=eu-central-2`. Bucket: `crawl-data-951752554117-eu-central-2-an`, prefix `prod/`.

**MCP split**: port 8000 = FastAPI data backend; port 8001 = thin MCP bridge with single `search_listings` tool + React widget (`apps_sdk/web/src/App.tsx`).

**SRED special case**: if `raw_data/SRED_data(1)/` exists, bootstrap auto-generates `raw_data/sred_data.csv` and flattens images to `raw_data/sred_images/`.
