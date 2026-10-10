from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..db import connect
from .engine import Engine
from .imaging import load_image
from .vector_store import VectorMeta

FLANKS = {"left", "right", "unknown"}


def now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def create_animal(c, species: str = "Tiger", sex: str | None = None) -> dict:
    cur = c.execute("INSERT INTO animals (species, sex, created_at) VALUES (?,?,?)", (species, sex, now()))
    code = f"T-{cur.lastrowid:03d}"
    c.execute("UPDATE animals SET code=? WHERE id=?", (code, cur.lastrowid))
    return {"id": code, "species": species, "sex": sex}


def save_image(engine: Engine, subdir: str, data: bytes, ext: str) -> str:
    sha = hashlib.sha256(data).hexdigest()
    rel = Path(subdir) / f"{sha[:16]}{ext}"
    path = engine.s.images_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return rel.as_posix()


def add_reference(engine: Engine, animal_code: str, data: bytes, flank: str, filename: str = "") -> dict:
    """Detect -> crop -> embed -> persist. Raises ValueError (bad input) or LookupError (no animal)."""
    if flank not in FLANKS:
        raise ValueError("flank must be left, right or unknown")
    img, _ = load_image(data)
    sha = hashlib.sha256(data).hexdigest()
    with connect(engine.s.db_path) as c:
        if not c.execute("SELECT 1 FROM animals WHERE code=?", (animal_code,)).fetchone():
            raise LookupError(f"Animal {animal_code} not found")
        dup = c.execute("SELECT id, animal_code FROM reference_images WHERE sha256=?", (sha,)).fetchone()
        if dup:
            return {"duplicate": True, "reference_id": dup["id"], "existing_animal": dup["animal_code"]}
    result = engine.analyze(img)
    if result is None:
        raise ValueError("No animal detected in the reference image")
    det, vec = result
    rel = save_image(engine, f"refs/{animal_code}", data, Path(filename).suffix.lower() or ".jpg")
    v = vec.astype("float32")
    with connect(engine.s.db_path) as c:
        cur = c.execute(
            "INSERT INTO reference_images (animal_code,image_path,flank,bbox,detection_confidence,sha256,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (animal_code, rel, flank, json.dumps(det.bbox), det.confidence, sha, now()),
        )
        ref_id = cur.lastrowid
        c.execute(
            "INSERT INTO embeddings (reference_id,animal_code,flank,vector,dim,model_name,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (ref_id, animal_code, flank, v.tobytes(), len(v), engine.embedder.model_id, now()),
        )
    engine.store.add(v, [VectorMeta(str(ref_id), animal_code, flank)])
    return {"duplicate": False, "reference_id": ref_id, "flank": flank, "detection_confidence": det.confidence}
