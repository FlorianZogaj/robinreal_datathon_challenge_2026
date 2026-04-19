"""
Image similarity scoring via pre-computed Jina CLIP v2 embeddings.

Loads data/image_embeddings.npz once on first use. Loads the Jina CLIP v2
text encoder locally (GPU if available, else CPU) for query-time embedding.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

_EMBEDDINGS_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "image_embeddings.npz"

# Module-level cache
_vectors_norm: np.ndarray | None = None
_listing_index: dict[str, list[int]] | None = None
_npz_loaded: bool = False

_model: Any = None
_model_loaded: bool = False


def _load_npz() -> tuple[np.ndarray | None, dict[str, list[int]]]:
    global _vectors_norm, _listing_index, _npz_loaded
    if _npz_loaded:
        return _vectors_norm, _listing_index or {}
    _npz_loaded = True
    if not _EMBEDDINGS_PATH.exists():
        return None, {}
    data = np.load(_EMBEDDINGS_PATH, allow_pickle=False)
    vectors = data["vectors"].astype(np.float32)
    _listing_index = json.loads(str(data["listing_index"]))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    _vectors_norm = vectors / np.maximum(norms, 1e-9)
    return _vectors_norm, _listing_index


def _load_model() -> Any:
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    try:
        import torch
        from transformers import AutoModel
        print("[image_search] loading jina-clip-v2 text encoder on cpu...")
        _model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True, dtype=torch.float32)
        _model.eval()
        print("[image_search] model ready on cpu")
    except Exception as e:
        print(f"[image_search] model load failed: {e}")
        _model = None
    return _model


def _embed_query(text: str) -> np.ndarray | None:
    model = _load_model()
    if model is None:
        return None
    try:
        import torch
        with torch.no_grad():
            vec = model.encode_text([text])
        if hasattr(vec, "cpu"):
            vec = vec.cpu().numpy()
        vec = np.array(vec, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-9)
    except Exception as e:
        print(f"[image_search] embed failed: {e}")
        return None


def image_similarity_scores(
    query_text: str,
    listing_ids: list[str],
    api_key: str = "",  # unused, kept for signature compatibility
) -> dict[str, float]:
    """
    Returns max cosine similarity between the query and each listing's images.
    Returns {} if embeddings file is missing or model fails to load.
    """
    vecs_norm, index = _load_npz()
    if vecs_norm is None:
        return {}
    q_vec = _embed_query(query_text)
    if q_vec is None:
        return {}
    scores: dict[str, float] = {}
    for lid in listing_ids:
        indices = index.get(str(lid))
        if indices:
            scores[str(lid)] = float((vecs_norm[indices] @ q_vec).max())
    return scores
