from __future__ import annotations

import threading
from dataclasses import dataclass

import faiss
import numpy as np

from .reid import Hit


@dataclass(frozen=True)
class VectorMeta:
    reference_id: str
    animal_id: str
    flank: str  # left | right | unknown


class VectorStore:
    """Exact cosine search (IndexFlatIP on L2-normalised vectors).
    Derived data: always rebuildable from the embeddings table."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._meta: list[VectorMeta] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return 0 if self._index is None else self._index.ntotal

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(v, dtype="float32")
        if v.ndim == 1:
            v = v[None, :]
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)

    def add(self, vectors: np.ndarray, metas: list[VectorMeta]) -> None:
        vectors = self.normalize(vectors)
        if len(metas) != len(vectors):
            raise ValueError("metadata length mismatch")
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexFlatIP(vectors.shape[1])
            elif vectors.shape[1] != self._index.d:
                raise ValueError("embedding dimension mismatch")
            self._index.add(vectors)
            self._meta.extend(metas)

    def rebuild(self, vectors: np.ndarray | None, metas: list[VectorMeta]) -> None:
        with self._lock:
            self._index, self._meta = None, []
        if metas:
            self.add(vectors, metas)

    def search(self, query: np.ndarray, flank: str | None = None) -> list[Hit]:
        """Gallery is small, so search everything and post-filter by flank."""
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            sims, idxs = self._index.search(self.normalize(query), self._index.ntotal)
            meta = list(self._meta)
        hits = []
        for sim, i in zip(sims[0], idxs[0]):
            m = meta[i]
            if flank in (None, "unknown") or m.flank in (flank, "unknown"):
                hits.append(Hit(m.animal_id, m.reference_id, float(sim)))
        return hits
