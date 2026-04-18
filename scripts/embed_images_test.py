"""
Test script for Voyage multimodal-3.5 image embedding pipeline.
Processes a small subset of listings (default 20) to validate the full flow:
  - fetches images from all source types (Comparis CDN, S3, local SRED)
  - embeds via Voyage API with input_type="document"
  - saves vectors + listing index to data/embeddings_test.npz
  - runs a sample text query to verify retrieval works

Usage:
  VOYAGE_API_KEY=your_key uv run python scripts/embed_images_test.py
  VOYAGE_API_KEY=your_key uv run python scripts/embed_images_test.py --n 50
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import sqlite3
import sys
import urllib.request
from typing import Any

import boto3
import numpy as np
import voyageai
from botocore.config import Config
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.config import get_settings

S3_RE = re.compile(r"https://([^.]+)\.s3[^/]*/(.+)")
SRED_IMAGES_DIR = pathlib.Path(__file__).resolve().parents[1] / "raw_data" / "sred_images"
MODEL = "voyage-multimodal-3.5"


def load_image(url: str, s3_client: Any) -> Image.Image | None:
    """Load a PIL Image from any source type. Returns None on failure."""
    # Local SRED image
    if url.startswith("/raw-data-images/"):
        path = SRED_IMAGES_DIR / pathlib.Path(url).name
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"    local load failed {path.name}: {e}")
            return None

    # S3 URL — presign before fetching
    if "s3.eu-central" in url or "crawl-data" in url:
        m = S3_RE.match(url)
        if not m:
            return None
        settings = get_settings()
        try:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.s3_bucket, "Key": m.group(2)},
                ExpiresIn=120,
            )
        except Exception as e:
            print(f"    presign failed: {e}")
            return None

    # HTTP fetch (Comparis CDN or presigned S3)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=10).read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"    fetch failed: {e}")
        return None


def embed_batch(client: voyageai.Client, images: list[Image.Image]) -> list[list[float]]:
    import time
    for attempt in range(4):
        try:
            result = client.multimodal_embed(
                inputs=[[img] for img in images],  # each input must be a list
                model=MODEL,
                input_type="document",
            )
            return result.embeddings
        except Exception as e:
            if "rate" in str(e).lower() and attempt < 3:
                time.sleep(20 * (attempt + 1))
                continue
            raise


def main(n: int, out_path: pathlib.Path) -> None:
    settings = get_settings()
    voyage = voyageai.Client()
    s3 = boto3.client(
        "s3",
        region_name=settings.s3_region,
        config=Config(s3={"addressing_style": "virtual"}),
    )

    db = settings.db_path
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        "SELECT listing_id, images_json FROM listings WHERE images_json IS NOT NULL ORDER BY RANDOM() LIMIT ?",
        [n],
    ).fetchall()

    all_vectors: list[list[float]] = []
    listing_index: dict[str, list[int]] = {}  # listing_id -> vector indices

    for row in rows:
        listing_id = row["listing_id"]
        try:
            imgs_data = json.loads(row["images_json"]).get("images", [])
        except Exception:
            continue

        urls = [
            (img.get("url", "") if isinstance(img, dict) else str(img))
            for img in imgs_data
        ]
        urls = [u for u in urls if u and "%" not in u]  # skip broken placeholder URLs

        print(f"[{listing_id}] {len(urls)} image(s)")

        pil_images: list[Image.Image] = []
        for url in urls:
            img = load_image(url, s3)
            if img is not None:
                pil_images.append(img)

        if not pil_images:
            print(f"  -> no usable images, skipping")
            continue

        try:
            vectors = embed_batch(voyage, pil_images)
        except Exception as e:
            print(f"  -> voyage embed failed: {e}")
            continue

        start = len(all_vectors)
        all_vectors.extend(vectors)
        listing_index[listing_id] = list(range(start, start + len(vectors)))
        print(f"  -> embedded {len(vectors)} image(s), dim={len(vectors[0])}")

    if not all_vectors:
        print("No vectors produced — check your API key and image sources.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        vectors=np.array(all_vectors, dtype=np.float32),
        listing_index=json.dumps(listing_index),
    )
    print(f"\nSaved {len(all_vectors)} vectors for {len(listing_index)} listings -> {out_path}")

    # Quick retrieval smoke test
    print("\n--- Retrieval smoke test ---")
    query = "bright modern apartment with large windows"
    print(f"Query: '{query}'")
    result = voyage.multimodal_embed(inputs=[[query]], model=MODEL, input_type="query")
    q_vec = np.array(result.embeddings[0], dtype=np.float32)

    vectors = np.array(all_vectors, dtype=np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / np.maximum(norms, 1e-9)
    q_norm = q_vec / max(np.linalg.norm(q_vec), 1e-9)

    # Max similarity per listing
    scores: list[tuple[float, str]] = []
    for lid, indices in listing_index.items():
        sims = vectors_norm[indices] @ q_norm
        scores.append((float(sims.max()), lid))

    scores.sort(reverse=True)
    print("Top 5 matches:")
    for score, lid in scores[:5]:
        print(f"  {lid}: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of listings to test")
    parser.add_argument("--out", type=str, default="data/embeddings_test.npz")
    args = parser.parse_args()
    main(args.n, pathlib.Path(args.out))
