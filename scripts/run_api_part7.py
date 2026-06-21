#!/usr/bin/env python3
"""
Part 7: Chạy FastAPI server.

Chạy sau Part 6:
    python scripts/run_api_part7.py

Test:
    python scripts/explore_part7.py
    bash scripts/test_api_part7.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def main() -> None:
    config = load_config()
    serving = config["serving"]
    host = serving["host"]
    port = int(serving["port"])

    print("=" * 60)
    print("Part 7: FastAPI Serving")
    print("=" * 60)
    print(f"Source: {serving.get('source', 'mlflow')} | stage={serving.get('model_stage', 'Staging')}")
    print(f"URL: http://{host}:{port}")
    print("Endpoints: GET /health | GET /model-info | POST /predict")
    print("Docs: http://127.0.0.1:8000/docs")
    print("=" * 60)

    uvicorn.run("src.serving.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
