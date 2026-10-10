from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import get_settings
from .db import connect, init_db
from .services import enroll
from .services.engine import Engine
from .services.imaging import load_image
from .services.reid import aggregate_by_animal, decide

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wildtrack")
STATIC = Path(__file__).parent / "static"
_engine: Engine | None = None

# Synthetic demo cameras (NOT real coordinates) - seeded on first run.
DEMO_CAMERAS = [(f"C-{i:03d}", 20.20 + 0.03 * (i % 4), 79.30 + 0.04 * (i // 4), z)
                for i, z in enumerate(["Core-A", "Core-B", "Buffer-A", "Buffer-B"] * 3, start=1)][:10]


def eng() -> Engine:
    if _engine is None:
        raise HTTPException(503, "Engine not ready")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    s = get_settings()
    init_db(s.db_path)
    s.images_dir.mkdir(parents=True, exist_ok=True)
    with connect(s.db_path) as c:
        if c.execute("SELECT COUNT(*) FROM cameras").fetchone()[0] == 0:
            c.executemany("INSERT INTO cameras (code,latitude,longitude,zone) VALUES (?,?,?,?)", DEMO_CAMERAS)
    _engine = Engine(s)
    log.info("Index rebuilt: %s", _engine.rebuild_index())
    yield


def require_key(x_api_key: str | None = Header(default=None)):
    if x_api_key != get_settings().api_key:
        raise HTTPException(401, "Invalid or missing API key")


app = FastAPI(title="WildTrack AI", lifespan=lifespan)
api = APIRouter(prefix="/api", dependencies=[Depends(require_key)])


# ---------- helpers ----------
def img_url(rel: str | None) -> str | None:
    return f"/images/{rel}" if rel else None


def camera_dict(r) -> dict:
    # Rounded (~1 km) coordinates for display: precise locations of protected wildlife are sensitive.
    return {"id": r["code"], "latitude": round(r["latitude"], 2), "longitude": round(r["longitude"], 2),
            "zone": r["zone"], "status": r["status"]}


def sighting_dict(c, r) -> dict:
    cands = json.loads(r["top_candidates"] or "[]")
    for cd in cands:
        ref = c.execute("SELECT image_path, flank FROM reference_images WHERE id=?", (cd.get("best_reference_id"),)).fetchone()
        cd["best_reference_image_url"] = img_url(ref["image_path"]) if ref else None
        cd["best_reference_flank"] = ref["flank"] if ref else None
    cam = c.execute("SELECT * FROM cameras WHERE code=?", (r["camera_code"],)).fetchone()
    return {
        "id": r["id"], "status": r["decision"], "review_status": r["review_status"],
        "animal_id": r["animal_code"], "predicted_animal_id": r["predicted_animal_code"],
        "camera": camera_dict(cam) if cam else {"id": r["camera_code"]},
        "image_url": img_url(r["image_path"]), "timestamp": r["timestamp"],
        "timestamp_source": r["timestamp_source"], "flank": r["flank"],
        "detection": {"confidence": r["detection_confidence"], "bbox": json.loads(r["bbox"]) if r["bbox"] else None},
        "similarity": r["similarity_score"], "margin": r["margin"], "top_candidates": cands,
        "model_version": json.loads(r["model_version"] or "{}"), "reviewed_at": r["reviewed_at"],
    }


# ---------- health ----------
@app.get("/api/health")
def health():
    e = eng()
    return {"status": "ok", "device": e.device, "detector": e.detector.name,
            "embedder": e.embedder.model_id, "index_size": e.store.size}


# ---------- animals ----------
class AnimalIn(BaseModel):
    species: str = "Tiger"
    sex: str | None = None


@api.post("/animals")
def create_animal(body: AnimalIn):
    with connect(eng().s.db_path) as c:
        return enroll.create_animal(c, body.species, body.sex)


def animal_summary(c, a) -> dict:
    code = a["code"]
    refs = c.execute("SELECT flank, COUNT(*) n FROM reference_images WHERE animal_code=? GROUP BY flank", (code,)).fetchall()
    cov = {"left": 0, "right": 0, "unknown": 0, **{r["flank"]: r["n"] for r in refs}}
    st = c.execute("SELECT COUNT(*) n, MIN(timestamp) f, MAX(timestamp) l FROM sightings "
                   "WHERE animal_code=? AND review_status='confirmed'", (code,)).fetchone()
    warnings = [f"No {side}-flank reference image" for side in ("left", "right") if cov[side] == 0]
    thumb = c.execute("SELECT image_path FROM reference_images WHERE animal_code=? LIMIT 1", (code,)).fetchone()
    return {"id": code, "species": a["species"], "sex": a["sex"], "status": a["status"],
            "flank_coverage": cov, "warnings": warnings, "sighting_count": st["n"],
            "first_seen": st["f"], "last_seen": st["l"], "thumbnail_url": img_url(thumb["image_path"]) if thumb else None}


@api.get("/animals")
def list_animals():
    with connect(eng().s.db_path) as c:
        return [animal_summary(c, a) for a in c.execute("SELECT * FROM animals ORDER BY id")]


@api.get("/animals/{animal_id}")
def get_animal(animal_id: str):
    with connect(eng().s.db_path) as c:
        a = c.execute("SELECT * FROM animals WHERE code=?", (animal_id,)).fetchone()
        if not a:
            raise HTTPException(404, "Animal not found")
        out = animal_summary(c, a)
        out["references"] = [
            {"id": r["id"], "flank": r["flank"], "image_url": img_url(r["image_path"])}
            for r in c.execute("SELECT * FROM reference_images WHERE animal_code=? ORDER BY id", (animal_id,))]
        return out


@api.post("/animals/{animal_id}/references")
def add_reference(animal_id: str, image: UploadFile = File(...), flank: str = Form("unknown")):
    e = eng()
    try:
        return enroll.add_reference(e, animal_id, image.file.read(), flank, image.filename or "")
    except LookupError as ex:
        raise HTTPException(404, str(ex))
    except ValueError as ex:
        raise HTTPException(400, str(ex))


@api.get("/animals/{animal_id}/sightings")
def animal_sightings(animal_id: str):
    with connect(eng().s.db_path) as c:
        rows = c.execute("SELECT * FROM sightings WHERE animal_code=? ORDER BY timestamp DESC", (animal_id,)).fetchall()
        return [sighting_dict(c, r) for r in rows]


# ---------- cameras ----------
class CameraIn(BaseModel):
    camera_code: str
    latitude: float
    longitude: float
    zone: str | None = None


@api.get("/cameras")
def list_cameras():
    with connect(eng().s.db_path) as c:
        return [camera_dict(r) for r in c.execute("SELECT * FROM cameras ORDER BY code")]


@api.post("/cameras")
def create_camera(body: CameraIn):
    with connect(eng().s.db_path) as c:
        try:
            c.execute("INSERT INTO cameras (code,latitude,longitude,zone) VALUES (?,?,?,?)",
                      (body.camera_code, body.latitude, body.longitude, body.zone))
        except Exception:
            raise HTTPException(409, "Camera already exists")
    return {"id": body.camera_code}


# ---------- identify ----------
@api.post("/identify")
def identify(image: UploadFile = File(...), camera_id: str = Form(...),
             timestamp: str | None = Form(None), flank: str | None = Form(None)):
    e = eng()
    if flank in ("", None):
        flank = None
    if flank not in (None, "left", "right", "unknown"):
        raise HTTPException(400, "flank must be left, right or unknown")
    data = image.file.read()
    try:
        img, exif_ts = load_image(data)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    if timestamp:
        try:
            ts, ts_src = datetime.fromisoformat(timestamp).replace(tzinfo=None), "request"
        except ValueError:
            raise HTTPException(400, "timestamp must be ISO 8601")
    elif exif_ts:
        ts, ts_src = exif_ts, "exif"
    else:
        ts, ts_src = datetime.now(), "server"

    with connect(e.s.db_path) as c:
        if not c.execute("SELECT 1 FROM cameras WHERE code=?", (camera_id,)).fetchone():
            raise HTTPException(400, f"Unknown camera {camera_id}")

    rel = enroll.save_image(e, "queries", data, Path(image.filename or "q.jpg").suffix.lower() or ".jpg")
    result = e.analyze(img)

    if result is None:
        status, sim, margin, cands, det_conf, bbox, pred, review_status = "no_detection", None, None, [], None, None, None, "auto"
    else:
        det, vec = result
        cands = aggregate_by_animal(e.store.search(vec, flank), e.s.top_k)
        d = decide(cands, e.thresholds)
        status, sim, margin = d.status, d.similarity, d.margin
        det_conf, bbox, review_status = det.confidence, list(det.bbox), "pending"
        pred = cands[0].animal_id if status in ("matched", "review") else None

    cand_json = [{"animal_id": x.animal_id, "similarity": round(x.similarity, 4),
                  "best_reference_id": int(x.best_reference_id)} for x in cands]
    with connect(e.s.db_path) as c:
        cur = c.execute(
            "INSERT INTO sightings (predicted_animal_code,camera_code,image_path,timestamp,timestamp_source,flank,bbox,"
            "detection_confidence,decision,similarity_score,margin,top_candidates,model_version,review_status,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pred, camera_id, rel, ts.isoformat(timespec="seconds"), ts_src, flank, json.dumps(bbox), det_conf, status,
             sim, margin, json.dumps(cand_json), json.dumps(e.model_version), review_status, enroll.now()))
        row = c.execute("SELECT * FROM sightings WHERE id=?", (cur.lastrowid,)).fetchone()
        out = sighting_dict(c, row)
    t = e.thresholds
    out.update({"sighting_id": out["id"], "thresholds": {"match": t.match, "review": t.review, "min_margin": t.min_margin},
                "image_size": [img.width, img.height],
                "animal": {"id": pred, "species": "Tiger"} if status == "matched" else None,
                "gallery_empty": e.store.size == 0})
    return out


# ---------- sightings ----------
@api.get("/sightings")
def list_sightings(review_status: str | None = None, animal_id: str | None = None, limit: int = 100):
    q, args = "SELECT * FROM sightings WHERE 1=1", []
    if review_status:
        q += " AND review_status=?"; args.append(review_status)
    if animal_id:
        q += " AND animal_code=?"; args.append(animal_id)
    q += " ORDER BY id DESC LIMIT ?"; args.append(min(limit, 500))
    with connect(eng().s.db_path) as c:
        return [sighting_dict(c, r) for r in c.execute(q, args)]


@api.get("/sightings/{sighting_id}")
def get_sighting(sighting_id: int):
    with connect(eng().s.db_path) as c:
        r = c.execute("SELECT * FROM sightings WHERE id=?", (sighting_id,)).fetchone()
        if not r:
            raise HTTPException(404, "Sighting not found")
        return sighting_dict(c, r)


class ReviewIn(BaseModel):
    action: str  # confirm | reject | reassign
    animal_id: str | None = None


@api.patch("/sightings/{sighting_id}")
def review_sighting(sighting_id: int, body: ReviewIn):
    with connect(eng().s.db_path) as c:
        r = c.execute("SELECT * FROM sightings WHERE id=?", (sighting_id,)).fetchone()
        if not r:
            raise HTTPException(404, "Sighting not found")
        if body.action == "confirm":
            if not r["predicted_animal_code"]:
                raise HTTPException(400, "No predicted animal to confirm; use reassign")
            animal, st = r["predicted_animal_code"], "confirmed"
        elif body.action == "reassign":
            if not body.animal_id or not c.execute("SELECT 1 FROM animals WHERE code=?", (body.animal_id,)).fetchone():
                raise HTTPException(400, "Valid animal_id required")
            animal, st = body.animal_id, "confirmed"
        elif body.action == "reject":
            animal, st = None, "rejected"
        else:
            raise HTTPException(400, "action must be confirm, reject or reassign")
        c.execute("UPDATE sightings SET animal_code=?, review_status=?, reviewed_at=? WHERE id=?",
                  (animal, st, enroll.now(), sighting_id))
        return sighting_dict(c, c.execute("SELECT * FROM sightings WHERE id=?", (sighting_id,)).fetchone())


# ---------- analytics / admin ----------
@api.get("/analytics/summary")
def summary():
    with connect(eng().s.db_path) as c:
        one = lambda q: c.execute(q).fetchone()[0]
        recent = c.execute("SELECT * FROM sightings ORDER BY id DESC LIMIT 6").fetchall()
        return {"animals": one("SELECT COUNT(*) FROM animals"), "cameras": one("SELECT COUNT(*) FROM cameras"),
                "sightings": one("SELECT COUNT(*) FROM sightings"),
                "pending": one("SELECT COUNT(*) FROM sightings WHERE review_status='pending'"),
                "confirmed": one("SELECT COUNT(*) FROM sightings WHERE review_status='confirmed'"),
                "reference_images": one("SELECT COUNT(*) FROM reference_images"),
                "recent": [sighting_dict(c, r) for r in recent]}


@api.get("/analytics/movement/{animal_id}")
def movement(animal_id: str):
    with connect(eng().s.db_path) as c:
        rows = c.execute(
            "SELECT s.id, s.timestamp, s.camera_code, c.latitude, c.longitude FROM sightings s "
            "JOIN cameras c ON c.code=s.camera_code WHERE s.animal_code=? AND s.review_status='confirmed' "
            "ORDER BY s.timestamp", (animal_id,)).fetchall()
        return [{"sighting_id": r["id"], "timestamp": r["timestamp"], "camera_id": r["camera_code"],
                 "latitude": round(r["latitude"], 2), "longitude": round(r["longitude"], 2)} for r in rows]


@api.post("/admin/index/rebuild")
def rebuild():
    return eng().rebuild_index()


app.include_router(api)
get_settings().images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=get_settings().images_dir), name="images")  # demo-only: public
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
