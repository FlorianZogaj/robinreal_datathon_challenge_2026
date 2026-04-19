"""
Colab/GPU image embedding pipeline using Jina CLIP v2.
Run this on Colab with an H100 (or any CUDA GPU).

Setup (run in Colab cell first):
    !pip install transformers torch pillow boto3 tqdm numpy python-dotenv

    # Upload your .env file and listings.db via the Colab file browser (left panel),
    # then set DB_PATH before running:
    import os; os.environ["DB_PATH"] = "/content/listings.db"

Usage:
    python embed_images_colab.py                  # full run
    python embed_images_colab.py --limit 500      # test run
    python embed_images_colab.py --resume         # resume from checkpoint

Output:
    image_embeddings.npz  — vectors (N, 1024) float32 + listing_index JSON
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import re
import sqlite3
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import boto3
import numpy as np
import torch
from botocore.config import Config
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

# Load credentials from .env if present (uploaded to /content/.env on Colab)
load_dotenv(pathlib.Path(os.environ.get("ENV_FILE", "/content/.env")))

MODEL_NAME = "jinaai/jina-clip-v2"
BATCH_SIZE = 128          # images per GPU forward pass — tune down if OOM
LISTING_BATCH = 50        # listings to fetch images for before embedding
CHECKPOINT_EVERY = 1000   # save progress every N listings processed
S3_RE = re.compile(r"https://([^.]+)\.s3[^/]*/(.+)")

OUT_PATH = pathlib.Path("image_embeddings.npz")
CHECKPOINT_PATH = pathlib.Path("image_embeddings_checkpoint.npz")

# Local SRED images (if mounted)
SRED_DIR = pathlib.Path(os.environ.get("SRED_DIR", "/content/sred_images"))


def load_image(url: str, s3_client: Any) -> Image.Image | None:
    if url.startswith("/raw-data-images/"):
        path = SRED_DIR / pathlib.Path(url).name
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    if "s3.eu-central" in url or "crawl-data" in url:
        m = S3_RE.match(url)
        if not m:
            return None
        bucket = os.environ.get("S3_BUCKET", "crawl-data-951752554117-eu-central-2-an")
        try:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": m.group(2)},
                ExpiresIn=300,
            )
        except Exception:
            return None

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=8).read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def collect_images(row: sqlite3.Row, s3_client: Any) -> tuple[str, list[Image.Image]]:
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

    pil_images: list[Image.Image | None] = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=min(len(urls), 8)) as ex:
        futures = {ex.submit(load_image, url, s3_client): i for i, url in enumerate(urls)}
        for f in as_completed(futures):
            pil_images[futures[f]] = f.result()

    return listing_id, [img for img in pil_images if img is not None]


def embed_images_gpu(model: AutoModel, images: list[Image.Image]) -> np.ndarray:
    """Embed images in GPU batches. Returns (N, 1024) float32 array."""
    all_vecs = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i : i + BATCH_SIZE]
        with torch.no_grad():
            vecs = model.encode_image(batch, batch_size=len(batch))
        if isinstance(vecs, torch.Tensor):
            vecs = vecs.cpu().numpy()
        all_vecs.append(np.array(vecs, dtype=np.float32))
    return np.concatenate(all_vecs, axis=0)


def embed_text_gpu(model: AutoModel, texts: list[str]) -> np.ndarray:
    """Embed text queries. Returns (N, 1024) float32 array."""
    with torch.no_grad():
        vecs = model.encode_text(texts)
    if isinstance(vecs, torch.Tensor):
        vecs = vecs.cpu().numpy()
    return np.array(vecs, dtype=np.float32)


def save_checkpoint(
    path: pathlib.Path,
    vectors: list[np.ndarray],
    listing_index: dict[str, list[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vectors=np.vstack(vectors) if vectors else np.zeros((0, 1024), dtype=np.float32),
        listing_index=json.dumps(listing_index),
    )


def load_checkpoint(path: pathlib.Path) -> tuple[list[np.ndarray], dict[str, list[int]]]:
    data = np.load(path, allow_pickle=False)
    vectors = [row for row in data["vectors"]]  # list of 1D arrays
    listing_index = json.loads(str(data["listing_index"]))
    return vectors, listing_index


def main(limit: int | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"Loading {MODEL_NAME}...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print("Model loaded.\n")

    db_path = os.environ.get("DB_PATH", "listings.db")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    query = "SELECT listing_id, images_json FROM listings WHERE images_json IS NOT NULL ORDER BY listing_id"
    if limit:
        query += f" LIMIT {limit}"
    rows = con.execute(query).fetchall()
    total = len(rows)
    print(f"Total listings: {total}")

    all_vectors: list[np.ndarray] = []
    listing_index: dict[str, list[int]] = {}

    if CHECKPOINT_PATH.exists():
        all_vectors, listing_index = load_checkpoint(CHECKPOINT_PATH)
        print(f"Resumed: {len(listing_index)} listings, {len(all_vectors)} vectors")

    already_done = set(listing_index.keys())
    remaining = [r for r in rows if r["listing_id"] not in already_done]
    print(f"Remaining: {len(remaining)} listings\n")

    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-central-2"),
        config=Config(s3={"addressing_style": "virtual"}),
    )

    processed = 0
    skipped = 0

    pbar = tqdm(
        remaining,
        desc="Embedding",
        unit="listing",
        initial=len(already_done),
        total=total,
    )

    for batch_start in range(0, len(remaining), LISTING_BATCH):
        batch_rows = remaining[batch_start : batch_start + LISTING_BATCH]

        # Fetch images for all listings in this batch concurrently
        batch: list[tuple[str, list[Image.Image]]] = []
        with ThreadPoolExecutor(max_workers=16) as ex:
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

        # Flatten all images, embed in GPU batches
        all_imgs_flat: list[Image.Image] = []
        sizes: list[int] = []
        for _, imgs in batch:
            all_imgs_flat.extend(imgs)
            sizes.append(len(imgs))

        vecs_flat = embed_images_gpu(model, all_imgs_flat)

        # Assign back to listings
        offset = 0
        for (lid, _), size in zip(batch, sizes):
            start = len(all_vectors)
            for vec in vecs_flat[offset : offset + size]:
                all_vectors.append(vec)
            listing_index[lid] = list(range(start, start + size))
            offset += size
            processed += 1
            pbar.update(1)

        pbar.set_postfix(vectors=len(all_vectors), skipped=skipped)

        if processed % CHECKPOINT_EVERY == 0 and processed > 0:
            save_checkpoint(CHECKPOINT_PATH, all_vectors, listing_index)
            print(f"\n  checkpoint saved ({len(listing_index)} listings)")

    # Final save
    save_checkpoint(OUT_PATH, all_vectors, listing_index)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\nDone. {len(all_vectors)} vectors for {len(listing_index)} listings -> {OUT_PATH}")
    print(f"processed={processed}  skipped={skipped}")

    # Quick smoke test
    print("\n--- Smoke test: embed a query ---")
    q = "bright modern apartment with large windows"
    q_vec = embed_text_gpu(model, [q])[0]
    vecs = np.vstack(all_vectors)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = vecs / np.maximum(norms, 1e-9)
    q_norm = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)

    scores: list[tuple[float, str]] = []
    for lid, indices in listing_index.items():
        sims = vecs_norm[indices] @ q_norm
        scores.append((float(sims.max()), lid))

    scores.sort(reverse=True)
    print(f"Top 5 for '{q}':")
    for score, lid in scores[:5]:
        print(f"  {lid}: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")  # auto by default
    args = parser.parse_args()
    main(args.limit)
