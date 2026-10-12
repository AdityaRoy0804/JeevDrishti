# 🐅 WildTrack AI

**From camera-trap images to persistent wildlife identities.**

WildTrack AI is a compact wildlife **individual re-identification (Re-ID)** prototype. It detects an animal in a camera-trap image, converts it into a visual embedding, searches a gallery of enrolled individuals, and returns a ranked match with evidence. If the evidence is weak, it says **unknown** or sends the case to **human review** instead of forcing an identity.

Built for the **WILDHACK 24-hour hackathon** (Problem Option A: *Re-identifying Individual Wild Animals*), with a tiger-focused demo.


---


## Features

| Area | Capability |
| --- | --- |
| **Detection** | MegaDetector (via PytorchWildlife) locates the animal; falls back to whole-image detection if the detector fails to load |
| **Embeddings** | MegaDescriptor (primary) or DINOv2 (fallback) via `timm`; L2-normalized vectors |
| **Search** | Exact cosine similarity with FAISS `IndexFlatIP`, aggregated per animal (max over reference images) |
| **Decisions** | `matched` / `review` / `unknown` / `no_detection` using match threshold, review threshold, and top-1/top-2 margin |
| **Enrollment** | Create animal profiles, upload multiple reference images tagged by flank (left / right / unknown), SHA-256 duplicate warning |
| **Flank handling** | Optional flank hint restricts search to same-flank vectors |
| **Human review** | Confirm, reject, or reassign every pending sighting |
| **Auditability** | Each sighting stores score, margin, top candidates, bounding box, timestamp source, and model versions |
| **Visualization** | Dashboard, animal registry, sighting review, and a Leaflet map of camera sightings |
| **Smoke-test mode** | Dummy embedder runs the whole app without downloading model weights |

---

## How It Works

```
Query image
    │
    ▼
Load (EXIF rotate, RGB, EXIF timestamp)
    │
    ▼
MegaDetector ──► no animal? ──► status = no_detection
    │
    ▼
Crop with 5% margin
    │
    ▼
Embedder (MegaDescriptor / DINOv2) ──► L2-normalized vector
    │
    ▼
FAISS search (optional flank filter)
    │
    ▼
Aggregate per animal (max similarity) ──► top-k candidates
    │
    ▼
Decide: matched | review | unknown
    │
    ▼
Store pending sighting in SQLite ──► human confirm / reject / reassign
```

**Enrollment** follows the same detect → crop → embed path. The vector is saved to SQLite and added to the in-memory FAISS index.

**Source of truth:** SQLite holds every embedding. The FAISS index is *derived* data and is rebuilt from the database at startup or through the admin endpoint, so the two can never drift apart.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend | Python, FastAPI |
| Data store | SQLite |
| Vector search | FAISS (`IndexFlatIP`) |
| Detection | PytorchWildlife (MegaDetector) |
| Embeddings | PyTorch + `timm` (MegaDescriptor / DINOv2) |
| Image processing | Pillow, NumPy |
| Frontend | Single-page HTML/JS (`app/static/index.html`, no build step) + Leaflet |
| Config | Environment variables via `.env` and `python-dotenv` |
| Testing | pytest |

---

## Project Structure

```
wildtrack-ai/
├── app/
│   ├── main.py              # FastAPI app, lifespan setup, /api routes, serves the UI at /
│   ├── config.py            # Settings loader (env vars, DB and image paths)
│   ├── db.py                # SQLite schema and connection manager
│   ├── services/
│   │   ├── detector.py      # MegaDetector wrapper + NoDetector fallback
│   │   ├── embedder.py      # Embedder factory: dummy / MegaDescriptor / DINOv2
│   │   ├── vector_store.py  # Thread-safe FAISS wrapper with flank filtering
│   │   ├── reid.py          # Per-animal aggregation and decision thresholds
│   │   ├── engine.py        # Orchestrates detector + embedder + vector store
│   │   ├── enroll.py        # Animal creation and reference-image enrollment
│   │   └── imaging.py       # EXIF handling and crop-with-margin
│   └── static/
│       └── index.html       # UI: Dashboard, Identify, Animals, Review, Map (+ vendored Leaflet)
├── scripts/
│   ├── seed_gallery.py      # Bulk-load gallery from the data/ folder tree
│   ├── evaluate.py          # Metrics and threshold sweep
│   └── build_atrw_subset.py # Dataset preprocessing
├── tests/
│   ├── conftest.py          # Fixtures using dummy models
│   ├── test_api.py          # API endpoint tests
│   └── test_reid.py         # Decision-logic tests
├── data/
│   ├── gallery/             # Enrolled animals, organized by animal and flank
│   ├── test/                # Held-out test queries
│   ├── unenrolled/          # Individuals never enrolled (unknown-state testing)
│   ├── images/              # Runtime storage (refs/, queries/)
│   ├── wildtrack.db         # SQLite database (auto-created)
│   └── LICENSES.md          # Source, license, and attribution for every image set
├── .env.example
├── requirements-core.txt    # FastAPI, FAISS, Pillow, NumPy, etc.
├── requirements.txt         # Core + torch, torchvision, timm, PytorchWildlife
├── run.sh / run.bat         # Startup scripts
└── README.md
```

### Gallery layout

```
data/gallery/
├── T-001/
│   ├── left/   ref_01.jpg ...
│   └── right/  ref_01.jpg ...
└── T-002/
    ├── left/
    └── right/
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- ~2 GB free disk space for model weights (full ML mode)
- A GPU is optional. CPU works, ideally with a lighter embedder such as `MegaDescriptor-B-224`.

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd wildtrack-ai

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

**Option A: Smoke-test mode** (no model weights, uses the dummy embedder):

```bash
pip install -r requirements-core.txt
```

**Option B: Full ML mode:**

```bash
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env             # Windows: copy .env.example .env
```

Edit `.env`. See [Configuration](#configuration).

### 4. Run

```bash
./run.sh                         # Windows: run.bat
```

Open the UI at **http://localhost:8000** (adjust if your `run` script uses another port).

### 5. Seed the demo gallery (optional)

```bash
python -m scripts.seed_gallery
```

> 💡 Pre-download and cache model weights before a live demo. Do not rely on venue internet for the first model load.

---

## Configuration

All settings are read from environment variables (`.env`). See `.env.example` for the full list.

| Variable | Purpose | Notes |
| --- | --- | --- |
| `API_KEY` | Operator key sent in the `X-API-Key` header | Set your own secret. Never commit it. |
| `DETECTOR` | Detector backend | MegaDetector, with whole-image fallback |
| `DETECTOR_MODEL` | MegaDetector variant | Use a variant name your installed PytorchWildlife version accepts |
| `EMBEDDER` | `megadescriptor`, `dinov2`, or `dummy` | `dummy` is for smoke tests only |
| `DATA_DIR` | Root for the database and image storage | |
| `MATCH_THRESHOLD` | Minimum top-1 similarity for `matched` | **Placeholder until calibrated** |
| `REVIEW_THRESHOLD` | Minimum top-1 similarity for `review` | **Placeholder until calibrated** |
| `MIN_MARGIN` | Minimum top-1 minus top-2 gap for `matched` | **Placeholder until calibrated** |

> Thresholds depend on the embedding model and dataset. Always recalibrate them after changing the embedder or gallery.

---

## Demo Flow

1. **Dashboard:** show enrolled animals, sightings, and cameras.
2. **Enroll:** create an animal and upload reference images with flank tags. Explain that the gallery is self-built.
3. **Identify:** upload a query image that was *not* used as a reference and comes from a different session.
4. **Inspect evidence:** detection box, candidate cards, reference-vs-query comparison, similarity, margin, decision badge.
5. **Review:** confirm the sighting and watch the profile and counts update.
6. **Map:** view where the individual was seen across cameras (camera coordinates are synthetic demo values).
7. **Unknown case:** upload an *unenrolled* tiger and show `unknown` or `review` rather than a forced match.
8. **Results:** show held-out evaluation numbers (see below).

---

## API Reference

All endpoints are under `/api` and require the `X-API-Key` header, except health.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service health, loaded models, index size |
| `POST` | `/api/animals` | Create an animal profile (`T-NNN` generated) |
| `GET` | `/api/animals` | List animals |
| `GET` | `/api/animals/{animal_id}` | Profile with flank coverage |
| `POST` | `/api/animals/{animal_id}/references` | Upload a reference image (`image`, `flank`) |
| `GET` | `/api/cameras` | List cameras |
| `POST` | `/api/cameras` | Register a camera |
| `POST` | `/api/identify` | Identify the animal in a query image; stores a pending sighting |
| `GET` | `/api/sightings` | List and filter sightings |
| `PATCH` | `/api/sightings/{sighting_id}` | `confirm`, `reject`, or `reassign` |
| `GET` | `/api/analytics/summary` | Dashboard metrics |
| `POST` | `/api/admin/index/rebuild` | Rebuild FAISS from stored embeddings |

> The exact set of routes is defined in `app/main.py`. Check it (or the auto-generated docs at `/docs`) for the current list.

### Example request

```bash
curl -X POST http://localhost:8000/api/identify \
  -H "X-API-Key: <your-key>" \
  -F "image=@data/test/query_01.jpg" \
  -F "camera_id=C-001" \
  -F "flank=left"
```

### Example response

```json
{
  "sighting_id": 42,
  "status": "matched",
  "animal": { "id": "T-001", "species": "Tiger" },
  "similarity": 0.876,
  "margin": 0.241,
  "top_candidates": [
    {
      "animal_id": "T-001",
      "similarity": 0.876,
      "best_reference_image_url": "/images/refs/T-001/ab12cd34.jpg",
      "best_reference_flank": "left"
    }
  ],
  "detection": { "confidence": 0.95, "bbox": [112, 86, 742, 612] },
  "thresholds": { "match": 0.80, "review": 0.65, "min_margin": 0.05 }
}
```

*All numeric values are illustrative.*

Possible `status` values: `matched`, `review`, `unknown`, `no_detection`.

---

## Database Schema

SQLite is the source of truth. Tables:

| Table | Key contents |
| --- | --- |
| `animals` | `animal_code` (e.g. `T-001`), species, sex, status, first/last seen |
| `cameras` | `camera_code`, latitude, longitude, zone, status (demo cameras are seeded) |
| `reference_images` | animal, image path, flank, bbox, detection confidence, `sha256` |
| `embeddings` | reference id, animal, flank, vector (float32 blob), dim, `model_name` |
| `sightings` | predicted and confirmed animal, camera, image, timestamp and source, flank, bbox, decision, similarity, margin, top candidates, model versions, `review_status`, `reviewed_at` |

**Review statuses:** `pending`, `auto` (no detection), `confirmed`, `rejected`.

Embeddings from different models are never mixed. On rebuild, vectors whose `model_name` does not match the active embedder are skipped.

---

## Decision Logic

Hits are aggregated per animal using the **maximum** similarity across that animal's reference images. Then:

| Status | Condition |
| --- | --- |
| `matched` | `top1 ≥ MATCH_THRESHOLD` **and** `(top1 − top2) ≥ MIN_MARGIN` |
| `review` | `top1 ≥ REVIEW_THRESHOLD` but the match conditions fail |
| `unknown` | `top1 < REVIEW_THRESHOLD` |
| `no_detection` | The detector finds no animal |

A tiger's left and right flanks have independent stripe patterns. A left-flank query cannot match an animal that has only right-flank references. Tag flanks at enrollment and enroll both sides where possible.

---

## Evaluation and Calibration

`scripts/evaluate.py` computes metrics and supports threshold sweeps on a held-out split.

Recommended protocol:

- **Split by session or camera event**, never at random, so near-duplicate frames do not appear in both gallery and queries.
- Tune thresholds on a **development** split. Report numbers only on a separate **test** split.
- Keep **unenrolled individuals** (`data/unenrolled/`) to measure unknown rejection and false-match rate.




---

## Testing

```bash
pytest
```

Tests use the dummy embedder, so no model weights are needed. Coverage includes API endpoints (`test_api.py`) and decision logic (`test_reid.py`).

---

## Design Decisions

| Decision | Reason |
| --- | --- |
| Pretrained models, no training | A 24-hour event is too short to train a Re-ID network; MegaDescriptor is built for individual re-ID |
| MegaDetector, not stock YOLO | Stock YOLO has no tiger class; MegaDetector is built for camera-trap animals |
| SQLite as source of truth, FAISS as derived index | Zero-setup storage; the index can always be rebuilt |
| `IndexFlatIP` on normalized vectors | Exact cosine search; fast enough at prototype scale |
| Margin check plus review band | Avoids confident wrong matches between similar-looking individuals |
| Pending sightings with human review | Model output is a suggestion, not a verdict |
| Single-file HTML frontend | No build step; simple to run and demo |
| Dummy embedder and detector fallback | The app stays runnable without weights |

---

## Known Limitations

- Prototype scale; not a population census tool.
- Embedding models are general Re-ID or vision models, not tiger-specific. Performance can drop with occlusion, blur, odd poses, or low resolution.
- Thresholds do not transfer across datasets or models.
- Flank is tagged by the operator; there is no automatic flank classification.
- MegaDetector detects animals, people, and vehicles but **not species**. The demo assumes the animal is a tiger.
- If MegaDetector fails to load, the app falls back to whole-image detection (a warning is logged).
- If the embedder fails to load, the app does not start (no automatic fallback).
- Camera coordinates are synthetic and are not evidence of real animal movement.
- Single operator API key; no user accounts or role-based access.
- MegaDescriptor weights are licensed **CC-BY-NC-4.0 (non-commercial)**. Review before any deployment.

---

## Responsible Use and Data Licenses

- Use only imagery you are legally permitted to use and redistribute.
- Record the source, license, and attribution of every image set in `data/LICENSES.md`.
- Do not publish sensitive wildlife locations; use rounded or offset coordinates in any public demo.
- Do not commit credentials, API keys, or private forest-department data.
- Treat model output as a candidate, not a certain identity. Keep a human in the loop for ambiguous cases.

---

## Roadmap

- Fine-tune a tiger-specific Re-ID model on authorized camera-trap data
- Automatic flank classification
- Additional species (leopard, bear, and others)
- Camera ingestion pipeline and edge inference
- Role-based access and secure data management
- Movement-path analytics and repeated-sighting alerts
- Integration with authorized forest-department systems

---

## Acknowledgements

- [MegaDetector / PytorchWildlife](https://github.com/microsoft/CameraTraps)
- [MegaDescriptor / WildlifeDatasets](https://github.com/WildlifeDatasets/wildlife-datasets)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [FAISS](https://github.com/facebookresearch/faiss)
- [ATRW dataset (LILA BC)](https://lila.science/)

