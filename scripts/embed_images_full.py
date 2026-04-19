"""
Full image embedding pipeline for all listings.
Embeds all images via Voyage multimodal-3.5 API and saves to data/image_embeddings.npz.

Features:
  - Processes all listings in the SQLite DB
  - Checkpoints every 500 listings — safe to interrupt and resume
  - Skips already-processed listings on resume
  - Handles all image sources: Comparis CDN, S3, local SRED
  - Gracefully skips 404s and other fetch errors

Usage:
  uv run python scripts/embed_images_full.py             # full run
  uv run python scripts/embed_images_full.py --limit 500 # partial run for testing
  uv run python scripts/embed_images_full.py --resume     # explicit resume (auto by default)

Output:
  data/image_embeddings.npz  — vectors (N, 1024) + listing_index JSON
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import sqlite3
import sys
import time
import urllib.request
from typing import Any

import boto3
import numpy as np
import voyageai
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.config import get_settings

S3_RE = re.compile(r"https://([^.]+)\.s3[^/]*/(.+)")
SRED_IMAGES_DIR = pathlib.Path(__file__).resolve().parents[1] / "raw_data" / "sred_images"
MODEL = "voyage-multimodal-3.5"
CHECKPOINT_EVERY = 500
BATCH_SIZE = 10  # listings per API request — smaller = more frequent progress updates
OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "image_embeddings.npz"
CHECKPOINT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "image_embeddings_checkpoint.npz"


def load_image(url: str, s3_client: Any) -> Image.Image | None:
    if url.startswith("/raw-data-images/"):
        path = SRED_IMAGES_DIR / pathlib.Path(url).name
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    if "s3.eu-central" in url or "crawl-data" in url:
        m = S3_RE.match(url)
        if not m:
            return None
        settings = get_settings()
        try:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.s3_bucket, "Key": m.group(2)},
                ExpiresIn=300,
            )
        except Exception:
            return None

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=5).read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def embed_images(client: voyageai.Client, images: list[Image.Image]) -> list[list[float]]:
    """Embed a flat list of images in one API request."""
    for attempt in range(5):
        try:
            result = client.multimodal_embed(
                inputs=[[img] for img in images],
                model=MODEL,
                input_type="document",
            )
            return result.embeddings
        except Exception as e:
            msg = str(e).lower()
            if ("rate" in msg or "429" in msg) and attempt < 4:
                wait = 30 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise


def collect_images(row: sqlite3.Row, s3_client: Any) -> tuple[str, list[Image.Image]]:
    """Load all usable images for a listing row in parallel. Returns (listing_id, images)."""
    listing_id = row["listing_id"]
    try:
        imgs_data = json.loads(row["images_json"]).get("images", [])
    except Exception:
        return listing_id, []

    urls = [
        (img.get("url", "") if isinstance(img, dict) else str(img))
        for img in imgs_data
    ]
    urls = [u for u in urls if u and "%" not in u]
    if not urls:
        return listing_id, []

    # Fetch all images for this listing in parallel
    pil_images: list[Image.Image | None] = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=min(len(urls), 4)) as ex:
        futures = {ex.submit(load_image, url, s3_client): i for i, url in enumerate(urls)}
        for f in as_completed(futures):
            pil_images[futures[f]] = f.result()

    return listing_id, [img for img in pil_images if img is not None]


def save_checkpoint(
    path: pathlib.Path,
    vectors: list[list[float]],
    listing_index: dict[str, list[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vectors=np.array(vectors, dtype=np.float32),
        listing_index=json.dumps(listing_index),
    )


def load_checkpoint(path: pathlib.Path) -> tuple[list[list[float]], dict[str, list[int]]]:
    data = np.load(path, allow_pickle=False)
    vectors = data["vectors"].tolist()
    listing_index = json.loads(str(data["listing_index"]))
    return vectors, listing_index


def main(limit: int | None) -> None:
    settings = get_settings()
    voyage = voyageai.Client()
    s3 = boto3.client(
        "s3",
        region_name=settings.s3_region,
        config=Config(s3={"addressing_style": "virtual"}),
    )

    con = sqlite3.connect(settings.db_path)
    con.row_factory = sqlite3.Row

    query = "SELECT listing_id, images_json FROM listings WHERE images_json IS NOT NULL ORDER BY listing_id"
    if limit:
        query += f" LIMIT {limit}"
    rows = con.execute(query).fetchall()
    total = len(rows)
    print(f"Total listings to process: {total}")

    # Resume from checkpoint if available
    all_vectors: list[list[float]] = []
    listing_index: dict[str, list[int]] = {}

    if CHECKPOINT_PATH.exists():
        all_vectors, listing_index = load_checkpoint(CHECKPOINT_PATH)
        print(f"Resumed from checkpoint: {len(listing_index)} listings, {len(all_vectors)} vectors")

    already_done = set(listing_index.keys())
    remaining = [r for r in rows if r["listing_id"] not in already_done]
    print(f"Remaining: {len(remaining)} listings\n")

    processed = 0
    skipped = 0
    failed = 0

    pbar = tqdm(remaining, desc="Embedding", unit="listing", initial=len(already_done), total=total)

    # Process in batches of BATCH_SIZE listings per API request
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch_rows = remaining[batch_start: batch_start + BATCH_SIZE]

        # Load images for all listings in the batch concurrently
        # Cap at 4 workers to avoid throttling Comparis CDN
        batch: list[tuple[str, list[Image.Image]]] = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(collect_images, row, s3): row for row in batch_rows}
            for f in as_completed(futures):
                lid, imgs = f.result()
                if imgs:
                    batch.append((lid, imgs))
                else:
                    skipped += 1
                    pbar.update(1)

        if not batch:
            continue

        # Flatten all images into one API request
        all_imgs_flat: list[Image.Image] = []
        sizes: list[int] = []  # number of images per listing
        for _, imgs in batch:
            all_imgs_flat.extend(imgs)
            sizes.append(len(imgs))

        try:
            all_vecs = embed_images(voyage, all_imgs_flat)
        except Exception as e:
            print(f"\n  batch embed failed: {e}")
            failed += len(batch)
            pbar.update(len(batch))
            continue

        # Assign vectors back to each listing
        offset = 0
        for (lid, _), size in zip(batch, sizes):
            start = len(all_vectors)
            all_vectors.extend(all_vecs[offset: offset + size])
            listing_index[lid] = list(range(start, start + size))
            offset += size
            processed += 1
            pbar.update(1)

        pbar.set_postfix(vectors=len(all_vectors), skipped=skipped, failed=failed)

        if processed % CHECKPOINT_EVERY == 0:
            save_checkpoint(CHECKPOINT_PATH, all_vectors, listing_index)
            print(f"\n  checkpoint saved ({len(listing_index)} listings)")

    # Final save
    save_checkpoint(OUT_PATH, all_vectors, listing_index)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\nDone. {len(all_vectors)} vectors for {len(listing_index)} listings -> {OUT_PATH}")
    print(f"processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Cap number of listings (default: all)")
    args = parser.parse_args()
    main(args.limit)
