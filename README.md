# WildTrack AI — Tiger Re-ID Prototype

Camera-trap image → **MegaDetector** crop → **MegaDescriptor** embedding → FAISS search over an *enrolled* gallery →
`matched` / `review` / `unknown` / `no_detection` → pending sighting → human review → profile + map.

Single FastAPI service (SQLite + FAISS) serving a single-page UI. No Postgres, Docker, or Node build step.
This is a prototype: it does **not** use any official Tadoba animal-ID database; IDs like `T-001` are internal.

## Quick start (GPU machine)

```bash
# 1. Install a CUDA build of PyTorch first: https://pytorch.org/get-started/locally/
# 2. Then:
./run.sh            # Linux/macOS      (Windows: run.bat)
# open http://localhost:8000
```
First run downloads MegaDetector and MegaDescriptor weights (internet needed once; they are cached afterwards —
**run once before the demo**, don't rely on venue Wi-Fi).

`./run.sh smoke` starts with a dummy embedder and no detector (no weights, no GPU) just to check the UI/API.
Dummy scores are meaningless.

## Demo data

Put permitted/licensed tiger images here, then run `python -m scripts.seed_gallery` (or enroll through the UI):

```
data/gallery/<name>/left/*.jpg     # tag flank by folder; files directly in <name>/ are "unknown" flank
data/gallery/<name>/right/*.jpg
data/test/<name>/*.jpg             # held-out known queries (different sessions than the gallery!)
data/unenrolled/<name>/*.jpg       # tigers never enrolled, for the "unknown" demo
```
Record sources/licenses in `data/LICENSES.md`. Split by camera event/session, never randomly.

## Calibrate before you trust it

Thresholds in `.env` are **placeholders**. Run:

```bash
python -m scripts.evaluate
```
It reports top-1/top-3, false-match rate on unenrolled tigers, latency, and a threshold sweep. Set
`MATCH_THRESHOLD`, `REVIEW_THRESHOLD`, `MIN_MARGIN` from a dev split; quote numbers only from a held-out split.
To compare models, run it once with `EMBEDDER=megadescriptor` and once with `EMBEDDER=dinov2`.
Embeddings from different models are not comparable: after switching, re-enroll (the index only loads vectors from the active model).

## Demo flow
Dashboard → Animals (create profile, upload left+right references) → Identify (pick a query image; show detection box,
reference-vs-query, score, margin) → Confirm → Animal profile updates → Map (movement path) → Identify an unenrolled tiger → `unknown`.

## Layout
```
app/main.py            REST API (/api/...) + serves UI       app/static/index.html   the whole UI
app/services/          detector, embedder, reid (decision rules), vector_store, enroll, engine
scripts/               seed_gallery.py, evaluate.py          tests/                  pytest (dummy models)
```
API docs: http://localhost:8000/docs (all `/api/*` routes need header `X-API-Key`, default `demo-key`; change in `.env`).
Tests: `pip install -r requirements-core.txt && pytest`

## Known limitations
- Model code paths (MegaDetector / MegaDescriptor loading) could not be exercised in the build sandbox; the logic, API, DB, UI
  and scripts are tested with dummy models. If MegaDetector fails to load, the app falls back to whole-image mode and
  `/api/health` (top-right of the UI) shows `none (fallback)`. Check the `DETECTOR_MODEL` string against your installed PytorchWildlife.
- Similarity is a model score, not a calibrated probability. Left and right flanks have independent stripes: a left-flank
  query cannot match an animal with only right-flank references.
- MegaDescriptor weights are CC-BY-NC-4.0 (non-commercial). Public re-ID datasets may overlap its training data.
- Demo cameras are synthetic coordinates; displayed coordinates are rounded. Images are served publicly and the API uses one shared key — prototype security only.
- Map tiles need internet (markers still show offline). Flank tagging is manual.
