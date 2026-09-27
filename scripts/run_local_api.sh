#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"

cd "$PROJECT_DIR"

export QDRANT_URL="http://127.0.0.1:6333"
export QDRANT_API_KEY=""
export QDRANT_COLLECTION="renovation_docs"
export QDRANT_ESTIMATES_COLLECTION="renovation_docs_estimates_test"

exec .venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
