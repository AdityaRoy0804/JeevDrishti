from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS animals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT UNIQUE,
  species TEXT NOT NULL DEFAULT 'Tiger',
  sex TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cameras (
  code TEXT PRIMARY KEY,
  latitude REAL NOT NULL,
  longitude REAL NOT NULL,
  zone TEXT,
  status TEXT NOT NULL DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS reference_images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  animal_code TEXT NOT NULL REFERENCES animals(code),
  image_path TEXT NOT NULL,
  flank TEXT NOT NULL DEFAULT 'unknown',
  bbox TEXT,
  detection_confidence REAL,
  sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
  reference_id INTEGER PRIMARY KEY REFERENCES reference_images(id) ON DELETE CASCADE,
  animal_code TEXT NOT NULL,
  flank TEXT NOT NULL,
  vector BLOB NOT NULL,
  dim INTEGER NOT NULL,
  model_name TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sightings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  animal_code TEXT REFERENCES animals(code),
  predicted_animal_code TEXT,
  camera_code TEXT NOT NULL REFERENCES cameras(code),
  image_path TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  timestamp_source TEXT NOT NULL,
  flank TEXT,
  bbox TEXT,
  detection_confidence REAL,
  decision TEXT NOT NULL,
  similarity_score REAL,
  margin REAL,
  top_candidates TEXT,
  model_version TEXT,
  review_status TEXT NOT NULL DEFAULT 'pending',
  reviewed_at TEXT,
  created_at TEXT NOT NULL
);
"""


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as c:
        c.executescript(SCHEMA)


@contextmanager
def connect(path: Path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
