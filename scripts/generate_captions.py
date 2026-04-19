#!/usr/bin/env python3
"""
Generate image captions for all listings using Gemini 2.5 Flash.

Usage:
    uv run python scripts/generate_captions.py

Requires GOOGLE_API_KEY in .env or environment.
Captions are saved to raw_data/image_captions.json and can be resumed
if interrupted — listings with existing non-empty captions are skipped,
empty ones are retried in case they failed due to rate limits.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import boto3
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "listings.db"
CAPTIONS_PATH = ROOT / "raw_data" / "image_captions.json"
SRED_IMAGES_DIR = ROOT / "raw_data" / "sred_images"

CONCURRENCY = 15
SAVE_INTERVAL = 500
MAX_RETRIES = 3

_S3_URL_RE = re.compile(r"https://([^.]+)\.s3[^/]*/(.+)")
_S3_BUCKET = "crawl-data-951752554117-eu-central-2-an"
_S3_REGION = "eu-central-2"

_PROMPT = (
    "Describe this real estate listing image in 2-3 sentences. "
    "Focus on: room type, brightness and lighting, floor and wall materials, "
    "notable features (balcony, garden, view, fireplace, modern appliances), "
    "and overall style (modern, rustic, renovated). Be specific and factual."
)


def _load_captions() -> dict[str, str]:
    return json.loads(CAPTIONS_PATH.read_text()) if CAPTIONS_PATH.exists() else {}


def _save_captions(captions: dict[str, str]) -> None:
    CAPTIONS_PATH.write_text(json.dumps(captions, ensure_ascii=False, indent=2))


def _get_listings() -> list[dict]:
    if not DB_PATH.exists():
        raise SystemExit(
            f"DB not found at {DB_PATH}. Start the server once first to build it."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT listing_id, scrape_source, images_json FROM listings"
        " WHERE images_json IS NOT NULL AND images_json != ''"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


_s3_client: boto3.client | None = None
_s3_lock = asyncio.Lock()


def _get_s3() -> boto3.client:
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=_S3_REGION)
    return _s3_client


def _fetch_image(row: dict) -> tuple[bytes, str] | None:
    scrape_source = (row.get("scrape_source") or "").upper()
    images_json = row.get("images_json") or ""

    if scrape_source == "SRED":
        listing_id = row["listing_id"]
        for ext in (".jpeg", ".jpg", ".png", ".webp"):
            path = SRED_IMAGES_DIR / f"{listing_id}{ext}"
            if path.exists():
                mime = "image/jpeg" if ext in (".jpeg", ".jpg") else f"image/{ext.lstrip('.')}"
                return path.read_bytes(), mime
        return None

    try:
        data = json.loads(images_json)
        images = data.get("images", []) if isinstance(data, dict) else data
        if not images:
            return None
        first = images[0]
        url = first.get("url") if isinstance(first, dict) else str(first)
    except (json.JSONDecodeError, IndexError, AttributeError, TypeError):
        return None

    if not url:
        return None

    ext = url.rsplit(".", 1)[-1].lower().split("?")[0] if "." in url else "jpeg"
    mime = "image/jpeg" if ext in ("jpeg", "jpg") else f"image/{ext}"

    m = _S3_URL_RE.match(url)
    if m:
        key = m.group(2)
        try:
            response = _get_s3().get_object(Bucket=_S3_BUCKET, Key=key)
            return response["Body"].read(), mime
        except Exception:
            return None

    # Plain HTTP URL (e.g. Comparis CDN)
    try:
        response = httpx.get(url, timeout=15, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type.startswith("image/"):
            mime = content_type
        return response.content, mime
    except Exception:
        return None


def _call_gemini(client: genai.Client, image_bytes: bytes, mime: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    _PROMPT,
                ],
            )
            return response.text.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < MAX_RETRIES - 1:
                wait = 30 * (attempt + 1)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded")


async def _caption_one(
    client: genai.Client,
    row: dict,
    captions: dict[str, str],
    sem: asyncio.Semaphore,
) -> None:
    listing_id = row["listing_id"]

    result = await asyncio.to_thread(_fetch_image, row)
    if not result:
        captions[listing_id] = ""
        return

    image_bytes, mime = result
    async with sem:
        try:
            caption = await asyncio.to_thread(_call_gemini, client, image_bytes, mime)
            captions[listing_id] = caption
        except Exception as e:
            print(f"  [warn] {listing_id}: {e}")
            captions[listing_id] = ""


async def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set — add it to .env or export it.")

    client = genai.Client(api_key=api_key)
    captions = _load_captions()
    rows = _get_listings()

    # Skip only listings that already have a non-empty caption
    pending = [r for r in rows if not captions.get(r["listing_id"])]

    already = len(rows) - len(pending)
    print(f"{len(rows)} listings | {already} already captioned | {len(pending)} pending")
    if not pending:
        print("Nothing to do.")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    done = [0]

    async def tracked(row: dict) -> None:
        await _caption_one(client, row, captions, sem)
        done[0] += 1
        if done[0] % SAVE_INTERVAL == 0:
            _save_captions(captions)
            print(f"  {done[0]}/{len(pending)} done...")

    await asyncio.gather(*[tracked(r) for r in pending])
    _save_captions(captions)
    print(f"Done — {len(captions)} captions saved to {CAPTIONS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
