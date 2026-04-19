#!/usr/bin/env python3
"""
Smoke-test for generate_captions.py — captions 5 listings (SRED + S3) and prints results.
Does NOT write to image_captions.json.

Usage:
    uv run python scripts/test_captions.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from google import genai

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from generate_captions import _fetch_image, _call_gemini, _get_listings

load_dotenv()

N = 5


async def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)
    rows = _get_listings()

    sred = [r for r in rows if (r.get("scrape_source") or "").upper() == "SRED"]
    other = [r for r in rows if (r.get("scrape_source") or "").upper() != "SRED"]
    sample = (sred[:3] + other[:2])[:N]

    print(f"Testing {len(sample)} listings ({len(sred[:3])} SRED, {len(other[:2])} S3)...\n")

    for row in sample:
        lid = row["listing_id"]
        src = row.get("scrape_source", "?")
        print(f"[{src}] {lid}")

        result = await asyncio.to_thread(_fetch_image, row)
        if not result:
            print("  -> no image found\n")
            continue

        image_bytes, mime = result
        print(f"  image: {len(image_bytes):,} bytes, {mime}")

        try:
            caption = await asyncio.to_thread(_call_gemini, client, image_bytes, mime)
            print(f"  caption: {caption}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
