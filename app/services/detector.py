from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

log = logging.getLogger("wildtrack.detector")


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float


class NoDetector:
    """Treats the whole image as the animal. Use only for smoke tests / pre-cropped data."""

    name = "none"

    def detect(self, img: Image.Image) -> Detection | None:
        return Detection((0, 0, img.width, img.height), 1.0)


class MegaDetector:
    """MegaDetector v6 via PytorchWildlife. Keeps the highest-confidence *animal* box (class 0)."""

    def __init__(self, version: str, conf: float, device: str):
        from PytorchWildlife.models import detection as pw_detection  # heavy import

        self.name = version
        self.conf = conf
        self.model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version=version)

    def detect(self, img: Image.Image) -> Detection | None:
        result = self.model.single_image_detection(np.array(img), det_conf_thres=self.conf)
        det = result["detections"]
        best: Detection | None = None
        for xyxy, conf, cls in zip(det.xyxy, det.confidence, det.class_id):
            if int(cls) != 0:  # 0 = animal; skip person / vehicle
                continue
            if best is None or float(conf) > best.confidence:
                best = Detection(tuple(int(v) for v in xyxy), float(conf))
        return best


def build_detector(settings, device: str):
    if settings.detector == "none":
        return NoDetector()
    try:
        return MegaDetector(settings.detector_model, settings.detection_conf, device)
    except Exception as e:  # keep the demo alive; /api/health shows the fallback
        log.warning("MegaDetector unavailable (%s). Falling back to whole-image mode.", e)
        d = NoDetector()
        d.name = "none (fallback)"
        return d
