import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

H = {"X-API-Key": "test-key"}


def png(seed: int, noise: float = 0.0) -> bytes:
    rng = np.random.RandomState(seed)
    a = rng.randint(0, 255, (128, 128, 3)).astype("float32")
    if noise:
        a += np.random.RandomState(seed + 999).normal(0, noise, a.shape)
    b = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype("uint8")).save(b, "PNG")
    return b.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def f(b, name="x.png"):
    return {"image": (name, b, "image/png")}


def test_auth_and_health(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/animals").status_code == 401


def test_full_flow(client):
    a = client.post("/api/animals", json={}, headers=H).json()["id"]
    r = client.post(f"/api/animals/{a}/references", files=f(png(1)), data={"flank": "left"}, headers=H).json()
    assert r["duplicate"] is False
    dup = client.post(f"/api/animals/{a}/references", files=f(png(1)), data={"flank": "left"}, headers=H).json()
    assert dup["duplicate"] is True

    ok = client.post("/api/identify", files=f(png(1, noise=5)), data={"camera_id": "C-001", "flank": "left"}, headers=H).json()
    assert ok["status"] == "matched" and ok["animal"]["id"] == a
    assert ok["top_candidates"][0]["best_reference_image_url"]

    conf = client.patch(f"/api/sightings/{ok['sighting_id']}", json={"action": "confirm"}, headers=H).json()
    assert conf["review_status"] == "confirmed" and conf["animal_id"] == a
    assert client.get(f"/api/animals/{a}", headers=H).json()["sighting_count"] == 1
    assert len(client.get(f"/api/analytics/movement/{a}", headers=H).json()) == 1

    unk = client.post("/api/identify", files=f(png(77)), data={"camera_id": "C-002"}, headers=H).json()
    assert unk["status"] == "unknown"
    assert client.patch(f"/api/sightings/{unk['sighting_id']}", json={"action": "confirm"}, headers=H).status_code == 400
    rej = client.patch(f"/api/sightings/{unk['sighting_id']}", json={"action": "reject"}, headers=H).json()
    assert rej["review_status"] == "rejected"


def test_flank_mismatch_not_matched(client):
    a = client.post("/api/animals", json={}, headers=H).json()["id"]
    client.post(f"/api/animals/{a}/references", files=f(png(5)), data={"flank": "right"}, headers=H)
    res = client.post("/api/identify", files=f(png(5, noise=3)), data={"camera_id": "C-001", "flank": "left"}, headers=H).json()
    assert all(c["animal_id"] != a for c in res["top_candidates"])


def test_errors_and_rebuild(client):
    bad = client.post("/api/identify", files={"image": ("x.jpg", b"not an image", "image/jpeg")},
                      data={"camera_id": "C-001"}, headers=H)
    assert bad.status_code == 400
    assert client.post("/api/identify", files=f(png(1)), data={"camera_id": "NOPE"}, headers=H).status_code == 400
    n = client.get("/api/health").json()["index_size"]
    assert client.post("/api/admin/index/rebuild", headers=H).json()["indexed"] == n
