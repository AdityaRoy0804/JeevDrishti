from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EMBEDDER_MODELS = {
    "megadescriptor": "hf-hub:BVRA/MegaDescriptor-L-384",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "dummy": "dummy-v1",
}


@dataclass(frozen=True)
class Settings:
    api_key: str
    data_dir: Path
    detector: str
    detector_model: str
    detection_conf: float
    embedder: str
    embedder_model: str
    device: str
    match_threshold: float
    review_threshold: float
    min_margin: float
    top_k: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "wildtrack.db"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"


@lru_cache
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    embedder = env("EMBEDDER", "megadescriptor").lower()
    data_dir = Path(env("DATA_DIR", "./data"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    return Settings(
        api_key=env("API_KEY", "demo-key"),
        data_dir=data_dir,
        detector=env("DETECTOR", "megadetector").lower(),
        detector_model=env("DETECTOR_MODEL", "MDV6-mit-yolov9-c"),
        detection_conf=float(env("DETECTION_CONFIDENCE", "0.2")),
        embedder=embedder,
        embedder_model=env("EMBEDDER_MODEL") or DEFAULT_EMBEDDER_MODELS.get(embedder, embedder),
        device=env("DEVICE", "auto").lower(),
        match_threshold=float(env("MATCH_THRESHOLD", "0.80")),
        review_threshold=float(env("REVIEW_THRESHOLD", "0.65")),
        min_margin=float(env("MIN_MARGIN", "0.05")),
        top_k=int(env("TOP_K", "5")),
    )
