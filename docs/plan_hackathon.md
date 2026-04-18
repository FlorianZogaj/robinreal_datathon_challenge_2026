# Hackathon Plan: Real-Estate Search & Ranking System

## Context
Starting from a FastAPI harness with 4 stub files. The challenge is to build a NL query → ranked listings pipeline for Swiss real-estate. Evaluation is on hard-filter precision (violations = disqualification) + ranking quality + breadth of query types handled. This plan delivers a working, competitive system as fast as possible.

## Critical Files to Modify

| File | Status | Action |
|------|--------|--------|
| `app/participant/hard_fact_extraction.py` | Stub | Implement with Claude API |
| `app/participant/soft_fact_extraction.py` | Stub | Implement via shared cache with above |
| `app/participant/ranking.py` | Stub | Implement multi-signal scoring + LLM re-rank |
| `app/participant/soft_filtering.py` | Stub | Implement conservative negative filter |
| `pyproject.toml` | Missing dep | Add `anthropic>=0.40.0,<1.0.0` |

**Do NOT modify:** harness, SQL layer, schemas, API routes, listing_row_parser.py.

---

## Implementation Order (Priority Order)

### Step 1 — Add anthropic dependency
```bash
uv add anthropic
```

Set env var: `export ANTHROPIC_API_KEY=sk-ant-...`

---

### Step 2 — hard_fact_extraction.py + soft_fact_extraction.py (Single Claude call)

Use ONE combined Claude call returning both `hard` and `soft` JSON. Cache the result at module level so the second call (soft_fact_extraction) reads from cache — avoiding double latency.

**Key design: Claude with `temperature=0`, strict JSON, safe fallback**

Combined system prompt extracts:
- **hard**: city, postal_code, canton, min/max_price, min/max_rooms, features, offer_type, object_category
- **soft**: brightness (0-1), modernity, quietness, spaciousness, views, family_friendly, commute_priority, value_priority, preferred_features[], negative_signals[], keywords[], inferred_price_sensitivity

**Swiss context rules the prompt must enforce:**
- "günstig"/"affordable" → `value_priority=0.8`, NOT a `max_price` constraint
- "ideally with parking" → `preferred_features`, NOT hard `features`
- "3.5 Zimmer" → `min_rooms=3.5, max_rooms=3.5` exactly
- "hell"/"lichtdurchflutet"/"bright" → `brightness=0.9`
- `keywords` should include German synonyms: bright→["hell","sonnig","lichtdurchflutet"]
- City aliases: Zurich/Zuerich → "Zürich", Geneva/Genf → "Genève"
- Safety: any parse failure returns `HardFilters()` (empty = returns all listings, never crashes)

**Cache pattern** (in hard_fact_extraction.py):
```python
_extraction_cache: dict[str, tuple[HardFilters, dict]] = {}

def extract_hard_facts(query: str) -> HardFilters:
    if query not in _extraction_cache:
        hard, soft = _call_claude(query)
        _extraction_cache.clear()  # keep only latest
        _extraction_cache[query] = (hard, soft)
    return _extraction_cache[query][0]
```

In soft_fact_extraction.py, import `_extraction_cache` and read the soft dict.

Use **prompt caching** (cache_control: ephemeral on system prompt) to reduce cost by ~90%.

---

### Step 3 — ranking.py (Multi-signal scoring + optional LLM re-rank)

**Scoring formula:**
```
score = 0.35 * text_score + 0.25 * feature_score + 0.20 * geo_score + 0.20 * soft_attr_score
```

**Signal 1 — text_score (BM25, ~30 lines, no external library)**
- Tokenize: `re.findall(r'\b\w+\b', text.lower())`
- Query terms from: `soft_facts["keywords"]` + tokenized `raw_query`
- Documents: `title + " " + description` for each candidate
- Compute per-term IDF over candidate set, then BM25 TF-IDF score
- Handles "hell", "ruhig", "modern" in listing descriptions naturally

**Signal 2 — feature_score**
- `wanted = set(soft_facts["preferred_features"])`
- Check both `feature_*` boolean DB columns AND listing's features list
- `score = matched_count / len(wanted)` — 0.5 neutral if no preferences

**Signal 3 — geo_score (uses existing DB columns)**
- `distance_public_transport`: score = `max(0, 1 - dist/1000)`, weighted by `commute_priority`
- `distance_kindergarten`, `distance_school_1`: weighted by `family_friendly`
- `distance_shop`: generic liveability signal (0.3 weight)

**Signal 4 — soft_attr_score (metadata proxies)**
- Brightness proxy: `feature_balcony=1` (+0.4) + `feature_new_build=1` (+0.3)
- Modernity proxy: `feature_new_build=1` (+0.5) + `feature_minergie_certified=1` (+0.3)
- Family proxy: `feature_child_friendly=1` (+0.5) + `distance_kindergarten < 400m` (+0.3)
- Value proxy: price/area ratio vs. Swiss benchmark (< 25 CHF/m² = good value)

**LLM re-rank (high-value, implement if time permits):**
- Pre-rank all candidates with formula above
- Send top 30 ONLY to Claude in a single batched prompt (NOT one call per listing)
- Each listing formatted as: `[ID] Title | CHF/mo | area | Features: X,Y | Description (200 chars)`
- Claude returns JSON array `[{listing_id, score, reason}]`
- Merge LLM scores with formula scores for final ordering

**Reason field:**
- For LLM re-ranked: use Claude's `reason`
- Otherwise: build from top signals, e.g. "Close to transport (120m). Balcony matched. Bright keywords in description."

---

### Step 4 — soft_filtering.py (Conservative, safe)

Only filter on strong negative evidence. **Never return empty list** — fallback to all candidates.

```python
def filter_soft_facts(candidates, soft_facts):
    if not soft_facts.get("negative_signals") and not soft_facts.get("inferred_price_sensitivity"):
        return candidates  # fast path
    
    result = [c for c in candidates
              if not any(neg.lower() in f"{c.get('title','')} {c.get('description','')}".lower()
                         for neg in soft_facts.get("negative_signals", []) if len(neg) > 4)]
    
    if soft_facts.get("inferred_price_sensitivity") == "high" and len(result) > 10:
        prices = sorted(c.get("price") or 0 for c in result)
        p80 = prices[int(len(prices) * 0.8)]
        result = [c for c in result if (c.get("price") or 0) <= p80]
    
    return result or candidates  # safety: never return empty
```

---

## Verification

1. Start server: `uv run uvicorn app.main:app --reload`
2. Test hard filtering:
   ```bash
   curl -X POST http://localhost:8000/listings \
     -H "content-type: application/json" \
     -d '{"query": "3.5-Zimmer Wohnung in Zürich unter 2800 CHF mit Balkon", "limit": 10}'
   ```
   Verify: all results have city=Zürich, rooms≈3.5, price≤2800, balcony feature
3. Test soft query: `"Bright family-friendly flat in Winterthur, not too expensive"`
   Verify: results sorted by brightness/family signals, no hard price constraint applied
4. Test Swiss German: `"Ich suche eine günstige Wohnung in Basel mit ÖV-Anschluss"`
   Verify: city=Basel, no price constraint, commute_priority high in soft_facts
5. Run existing tests: `uv run pytest tests -q`
6. Check `meta` in response for extracted filters (add to search_service.py `meta` dict for debugging)

---

## Bonus Enhancements (if time remains)

1. **Geospatial enrichment**: If query mentions landmark (ETH, HB, EPFL), geocode via Nominatim and set `latitude/longitude/radius_km` in HardFilters
2. **Image analysis**: Pass listing image URLs to Claude vision API for brightness/modernity scoring — add to soft_attr_score
3. **Commute time via SBB**: Use SBB API to compute actual transit times from listing to mentioned destination
4. **Multi-turn**: Add conversation history to the Claude prompt for session-aware ranking
5. **User preference profile**: Store interaction signals in SQLite, inject as context into ranking prompt

## Architecture Notes
- DB has ~4,900 rows; after hard filtering typically 10-200 candidates — BM25 is fast
- Single Claude call per query (combined hard+soft extraction) — ~500-800ms latency
- LLM re-rank adds ~300ms for top 30; total <1.5s per request
- Prompt caching on system prompt saves ~90% token cost on the system prompt
