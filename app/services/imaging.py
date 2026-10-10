from __future__ import annotations

import io
from datetime import datetime

from PIL import Image, ImageOps, UnidentifiedImageError


def load_image(data: bytes) -> tuple[Image.Image, datetime | None]:
    """Decode bytes -> RGB image (EXIF-rotated) and EXIF capture time if present."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError("Invalid or unsupported image file") from e
    ts = None
    try:
        exif = img.getexif()
        raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)
        if raw:
            ts = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
    except Exception:
        ts = None
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img, ts


def crop_with_margin(img: Image.Image, bbox: tuple[int, int, int, int], margin: float = 0.05) -> Image.Image:
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    box = (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(img.width, int(x2 + mx)), min(img.height, int(y2 + my)))
    return img.crop(box)
