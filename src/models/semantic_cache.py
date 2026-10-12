"""Embedding retrieval and append-only FIFO admission; labels never affect lookup."""
from dataclasses import dataclass
import numpy as np

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class SentenceEncoder:
    def __init__(self, model=MODEL, revision=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, revision=revision)

    def encode(self, query):
        return self.model.encode(query.strip(), normalize_embeddings=True)


def unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
        raise ValueError('Encoder must return a finite, nonzero one-dimensional embedding')
    return value / np.linalg.norm(value)


def cosine(a, b):
    return float(np.clip(np.dot(unit(a), unit(b)), -1, 1))


@dataclass
class Entry:
    query: str
    embedding: np.ndarray
    response: object
    intent_id: object
    insertion_order: int


class FIFOCache:
    def __init__(self, capacity=32, threshold=0.85):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        if not np.isfinite(threshold) or not -1 < threshold <= 1:
            raise ValueError('threshold must be in (-1, 1]')
        self.capacity, self.threshold = capacity, threshold
        self.entries, self.next_order = [], 0

    def lookup(self, embedding):
        unit(embedding)
        if not self.entries:
            return None, None
        scores = [cosine(embedding, e.embedding) for e in self.entries]
        index = int(np.argmax(scores))  # FIFO tie break
        return (self.entries[index], scores[index]) if scores[index] >= self.threshold else (None, None)

    def admit(self, query, embedding, response, intent_id=None):
        vector = unit(embedding)
        if self.entries and vector.shape != self.entries[0].embedding.shape:
            raise ValueError('Embedding dimensions must remain fixed')
        if len(self.entries) == self.capacity:
            self.entries.pop(0)
        self.entries.append(Entry(query, vector, response, intent_id, self.next_order))
        self.next_order += 1
