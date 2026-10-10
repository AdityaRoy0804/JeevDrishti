#!/usr/bin/env bash
# Usage: ./run.sh            (full models)   |   ./run.sh smoke   (no model weights; UI/API check only)
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
[ -f .env ] || cp .env.example .env
if [ "$1" = "smoke" ]; then
  pip install -q -r requirements-core.txt
  export EMBEDDER=dummy DETECTOR=none
else
  pip install -q -r requirements.txt
fi
uvicorn app.main:app --host 0.0.0.0 --port 8000
