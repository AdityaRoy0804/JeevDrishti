@echo off
REM Usage: run.bat            (full models)   |   run.bat smoke   (no model weights)
cd /d %~dp0
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
if not exist .env copy .env.example .env
if "%1"=="smoke" (
  pip install -q -r requirements-core.txt
  set EMBEDDER=dummy
  set DETECTOR=none
) else (
  pip install -q -r requirements.txt
)
uvicorn app.main:app --host 0.0.0.0 --port 8000
