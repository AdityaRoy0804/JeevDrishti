"""Enroll animals from a folder tree.

    data/gallery/<name>/left/*.jpg
    data/gallery/<name>/right/*.jpg
    data/gallery/<name>/*.jpg          (flank = unknown)

Each <name> folder becomes one animal (system IDs T-001, T-002, ... are assigned in order).
Run:  python -m scripts.seed_gallery
"""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.db import connect, init_db
from app.main import DEMO_CAMERAS
from app.services import enroll
from app.services.engine import Engine

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    s = get_settings()
    init_db(s.db_path)
    with connect(s.db_path) as c:
        if c.execute("SELECT COUNT(*) FROM cameras").fetchone()[0] == 0:
            c.executemany("INSERT INTO cameras (code,latitude,longitude,zone) VALUES (?,?,?,?)", DEMO_CAMERAS)
    e = Engine(s)
    gallery = s.data_dir / "gallery"
    folders = sorted(p for p in gallery.iterdir() if p.is_dir()) if gallery.exists() else []
    if not folders:
        print(f"No animal folders found in {gallery}. See README for the layout.")
        return
    for folder in folders:
        with connect(s.db_path) as c:
            animal = enroll.create_animal(c)
        added = dup = failed = 0
        for img in sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS):
            flank = img.parent.name if img.parent.name in ("left", "right") else "unknown"
            try:
                r = enroll.add_reference(e, animal["id"], img.read_bytes(), flank, img.name)
                dup += r["duplicate"]; added += not r["duplicate"]
            except ValueError as ex:
                failed += 1; print(f"  skip {img.name}: {ex}")
        print(f"{folder.name} -> {animal['id']}: {added} added, {dup} duplicate, {failed} failed")


if __name__ == "__main__":
    main()
