from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from ..config import Settings
from ..db import connect
from .detector import Detection, build_detector
from .embedder import build_embedder, resolve_device
from .imaging import crop_with_margin
from .reid import Thresholds
from .vector_store import VectorMeta, VectorStore

log = logging.getLogger("wildtrack.engine")


class Engine:
    """Holds the detector, embedder and in-memory vector index."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.device = resolve_device(settings.device)
        self.detector = build_detector(settings, self.device)
        self.embedder = build_embedder(settings, self.device)
        self.store = VectorStore()
        self.thresholds = Thresholds(settings.match_threshold, settings.review_threshold, settings.min_margin)

    @property
    def model_version(self) -> dict:
        return {"detector": self.detector.name, "embedder": self.embedder.model_id}

    def analyze(self, img: Image.Image) -> tuple[Detection, np.ndarray] | None:
        det = self.detector.detect(img)
        if det is None:
            return None
        return det, self.embedder.embed(crop_with_margin(img, det.bbox))

    def rebuild_index(self) -> dict:
        """The database is the source of truth; the index is derived from it."""
        with connect(self.s.db_path) as c:
            rows = c.execute("SELECT * FROM embeddings").fetchall()
        use = [r for r in rows if r["model_name"] == self.embedder.model_id]
        metas = [VectorMeta(str(r["reference_id"]), r["animal_code"], r["flank"]) for r in use]
        vecs = np.stack([np.frombuffer(r["vector"], dtype="float32") for r in use]) if use else None
        self.store.rebuild(vecs, metas)
        skipped = len(rows) - len(use)
        if skipped:
            log.warning("%d embeddings from other models were skipped; re-enroll to use them.", skipped)
        return {"indexed": len(use), "skipped_other_model": skipped}
