"""Embeddings e(q) for the query set Q (paper Sec. II-A; Sec. VI uses a 384-d
sentence-transformer). We use sentence-transformers/all-MiniLM-L6-v2, the same
encoder as the SemanticCacheController project, so both see the same vectors.

Embeddings are cached in an .npz next to the data, keyed by model name and the
exact query texts, so later runs are reproducible and need no model download.
"""

import os

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_queries(texts, cache_path, model=MODEL):
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["model"]) == model and list(cached["texts"]) == list(texts):
            return cached["embeddings"]

    from sentence_transformers import SentenceTransformer
    embeddings = SentenceTransformer(model).encode(list(texts), convert_to_numpy=True)
    np.savez(cache_path, model=model, texts=np.array(texts), embeddings=embeddings)
    return embeddings
