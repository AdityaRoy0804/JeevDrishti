"""Build a small demo dataset from the ATRW Re-ID *train* download (LILA BC).

Inputs : the Re-ID train annotations CSV (ID,filename pairs) and the folder with the Re-ID train images.
Outputs: data/gallery/<tiger_ID>/*.jpg     enrolled references   (flank = unknown; tag by hand later if you want)
         data/test/<tiger_ID>/*.jpg        held-out queries for the SAME tigers
         data/unenrolled/<tiger_ID>/*.jpg  tigers that are never enrolled (open-set test)
         data/LICENSES.md                  attribution

Usage:
  python -m scripts.build_atrw_subset --csv D:\\atrw\\reid_train.csv --images D:\\atrw\\reid_train_images --clean

Split: per tiger, a seeded random 60/40 split into gallery/test. In the real ATRW files the numbers in the file names of
one tiger are scattered (e.g. 105, 187, 201, 269 ...), so file names give no way to keep frames from the same video clip
on one side. Frames from the same clip can therefore end up in both gallery and test, which makes accuracy look better
than it would on genuinely new sessions. Say this when you present numbers.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from app.config import get_settings

EXTS = (".jpg", ".jpeg", ".png")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_pairs(csv_path: Path) -> list[tuple[str, str]]:
    """Return (tiger_id, filename) pairs; tolerant of column order, delimiter and a header row."""
    pairs = []
    for line in csv_path.read_text(encoding="utf-8-sig").splitlines():
        cols = [c for c in re.split(r"[,;\s]+", line.strip()) if c]
        if len(cols) < 2:
            continue
        a, b = cols[0], cols[1]
        if a.lower().endswith(EXTS):
            fname, tid = a, b
        elif b.lower().endswith(EXTS):
            fname, tid = b, a
        else:
            continue  # header or unknown row
        pairs.append((tid, Path(fname).name))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--enrolled", type=int, default=5)
    ap.add_argument("--unenrolled", type=int, default=3)
    ap.add_argument("--min-images", type=int, default=8, help="tigers with fewer images are ignored")
    ap.add_argument("--max-per-id", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clean", action="store_true", help="empty data/gallery, data/test, data/unenrolled first")
    a = ap.parse_args()

    data = get_settings().data_dir
    index = {p.name: p for p in a.images.rglob("*") if p.suffix.lower() in EXTS}
    by_id: dict[str, list[str]] = defaultdict(list)
    for tid, fname in read_pairs(a.csv):
        if fname in index:
            by_id[tid].append(fname)
    usable = sorted((t for t, f in by_id.items() if len(f) >= a.min_images), key=natural_key)
    print(f"{len(index)} images found, {len(by_id)} tigers matched to files, {len(usable)} with >= {a.min_images} images")
    need = a.enrolled + a.unenrolled
    if len(usable) < need:
        raise SystemExit(f"Need {need} tigers but only {len(usable)} are usable. Lower --min-images or the counts.")

    if a.clean:
        for sub in ("gallery", "test", "unenrolled"):
            shutil.rmtree(data / sub, ignore_errors=True)
    rng = random.Random(a.seed)
    chosen = rng.sample(usable, need)

    def copy(files: list[str], dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(index[f], dest / f)

    for tid in chosen[: a.enrolled]:
        files = sorted(by_id[tid], key=natural_key)
        files = rng.sample(files, min(len(files), a.max_per_id))
        g = max(1, int(len(files) * 0.6))
        copy(files[:g], data / "gallery" / f"tiger_{tid}")
        copy(files[g:], data / "test" / f"tiger_{tid}")
        print(f"enrolled   tiger_{tid}: {g} gallery / {len(files[g:])} test")
    for tid in chosen[a.enrolled:]:
        files = sorted(by_id[tid], key=natural_key)
        files = rng.sample(files, min(len(files), 4))
        copy(files, data / "unenrolled" / f"tiger_{tid}")
        print(f"unenrolled tiger_{tid}: {len(files)} images")

    (data / "LICENSES.md").write_text(
        "# Data sources\n\n"
        "## Amur Tiger Re-identification (ATRW), via LILA BC\n"
        "- Licence: CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution required)\n"
        "- Images owned by MakerCollider and the World Wildlife Fund\n"
        "- Content: video-derived images of Amur tigers from Chinese zoos (captive animals, not wild camera traps)\n"
        "- Citation: Li, S., Li, J., Lin, W., & Tang, H. (2019). Amur Tiger Re-identification in the Wild. arXiv:1906.05586\n"
        "- Note: MegaDescriptor may have been trained on data that overlaps this dataset; results can look better than on new data.\n",
        encoding="utf-8",
    )
    print("Done. Next: delete data/wildtrack.db and data/images, then python -m scripts.seed_gallery")


if __name__ == "__main__":
    main()