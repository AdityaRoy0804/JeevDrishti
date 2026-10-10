"""Offline evaluation + threshold sweep (no database needed).

Layout:
    data/gallery/<animal>/[left|right/]*.jpg     enrolled references
    data/test/<animal>/*.jpg                      known-animal queries (same <animal> names as gallery)
    data/unenrolled/<anything>/*.jpg              tigers never enrolled (open-set test)

IMPORTANT: split by session/camera event, not at random, and tune on a dev split but report on a held-out split.
Run:  python -m scripts.evaluate [--gallery DIR] [--test DIR] [--unenrolled DIR]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import get_settings
from app.services.engine import Engine
from app.services.imaging import load_image
from app.services.reid import Hit, Thresholds, aggregate_by_animal, decide
from app.services.vector_store import VectorMeta, VectorStore

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def files(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS) if root.exists() else []


def embed(e: Engine, path: Path):
    img, _ = load_image(path.read_bytes())
    t = time.perf_counter()
    r = e.analyze(img)
    return (None if r is None else r[1]), time.perf_counter() - t


def top_candidates(vs: VectorStore, vec, k=5):
    return aggregate_by_animal(vs.search(vec), k)


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=s.data_dir / "gallery", type=Path)
    ap.add_argument("--test", default=s.data_dir / "test", type=Path)
    ap.add_argument("--unenrolled", default=s.data_dir / "unenrolled", type=Path)
    a = ap.parse_args()
    e = Engine(s)

    vs = VectorStore()
    for p in files(a.gallery):
        rel = p.relative_to(a.gallery).parts
        flank = rel[1] if len(rel) > 2 and rel[1] in ("left", "right") else "unknown"
        v, _ = embed(e, p)
        if v is not None:
            vs.add(v, [VectorMeta(str(p), rel[0], flank)])
    print(f"Gallery vectors: {vs.size} | embedder: {e.embedder.model_id} | detector: {e.detector.name}")
    if vs.size == 0:
        return print("Empty gallery - nothing to evaluate.")

    known, unk, lat = [], [], []
    for p in files(a.test):
        v, dt = embed(e, p); lat.append(dt)
        if v is not None:
            known.append((p.relative_to(a.test).parts[0], top_candidates(vs, v)))
    for p in files(a.unenrolled):
        v, dt = embed(e, p); lat.append(dt)
        if v is not None:
            unk.append(top_candidates(vs, v))

    t = Thresholds(s.match_threshold, s.review_threshold, s.min_margin)
    if known:
        top1 = np.mean([c[0].animal_id == y for y, c in known if c])
        top3 = np.mean([y in [x.animal_id for x in c[:3]] for y, c in known])
        decs = [(y, decide(c, t)) for y, c in known]
        wrong_match = np.mean([d.status == "matched" and d.top_candidates[0].animal_id != y for y, d in decs])
        print(f"\nKnown queries: {len(known)} | top-1 {top1:.1%} | top-3 {top3:.1%}")
        print(f"At configured thresholds {t}: matched {np.mean([d.status=='matched' for _, d in decs]):.1%}, "
              f"review {np.mean([d.status=='review' for _, d in decs]):.1%}, "
              f"unknown {np.mean([d.status=='unknown' for _, d in decs]):.1%}, wrong-match {wrong_match:.1%}")
    if unk:
        ud = [decide(c, t) for c in unk]
        print(f"Unenrolled queries: {len(unk)} | false-match rate {np.mean([d.status=='matched' for d in ud]):.1%} | "
              f"unknown {np.mean([d.status=='unknown' for d in ud]):.1%} | review {np.mean([d.status=='review' for d in ud]):.1%}")
    if lat:
        print(f"Mean latency (detect+embed): {np.mean(lat)*1000:.0f} ms/image")

    if known:
        correct = np.array([c[0].similarity for y, c in known if c and c[0].animal_id == y])
        wrong = [c[0].similarity for y, c in known if c and c[0].animal_id != y] + [c[0].similarity for c in unk if c]
        print("\nSweep (match threshold on top-1 similarity, margin ignored):")
        print("thr    correct-accepted  false-accepted(wrong+unenrolled)")
        grid = np.linspace(min(np.min(correct) if len(correct) else 0, 0.0), 1.0, 21)
        for g in grid:
            fa = np.mean(np.array(wrong) >= g) if wrong else float("nan")
            print(f"{g:.2f}   {np.mean(correct >= g) if len(correct) else float('nan'):>14.1%}   {fa:>14.1%}")
        if wrong and len(correct):
            m = round(float(np.max(wrong)) + 0.01, 2)
            r = round(float(np.percentile(correct, 10)), 2)
            print(f"\nSuggested starting point: MATCH_THRESHOLD={m}  REVIEW_THRESHOLD={min(r, m - 0.05):.2f}")
            print("(Tune on a dev split; report numbers only from a separate held-out split.)")


if __name__ == "__main__":
    main()
