from __future__ import annotations

import threading

import numpy as np
from PIL import Image


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class DummyEmbedder:
    """No-weights embedder (centred 16x16 grayscale). Smoke tests only - NOT a real Re-ID model."""

    def __init__(self, model_id: str = "dummy-v1"):
        self.name = "dummy"
        self.model_id = model_id

    def embed(self, img: Image.Image) -> np.ndarray:
        a = np.asarray(img.convert("L").resize((16, 16)), dtype="float32").ravel()
        a -= a.mean()
        return a / (np.linalg.norm(a) + 1e-9)


class TimmEmbedder:
    """MegaDescriptor or DINOv2 through timm; pooled features, L2-normalised."""

    def __init__(self, name: str, model_id: str, device: str):
        import timm
        import torch

        self.name, self.model_id, self.device = name, model_id, device
        self._torch = torch
        self.model = timm.create_model(model_id, num_classes=0, pretrained=True).to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)
        self._lock = threading.Lock()  # one GPU forward at a time

    def embed(self, img: Image.Image) -> np.ndarray:
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        with self._lock, self._torch.inference_mode():
            f = self.model(x)
        v = f[0].float().cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-12)


def build_embedder(settings, device: str):
    if settings.embedder == "dummy":
        return DummyEmbedder(settings.embedder_model)
    if settings.embedder in ("megadescriptor", "dinov2"):
        return TimmEmbedder(settings.embedder, settings.embedder_model, device)
    raise ValueError(f"Unknown EMBEDDER '{settings.embedder}'")
