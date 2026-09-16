#!/bin/bash

# ==============================================================================
#  MY AI — Python Vector Database & LangGraph Platform Launcher
#  Python 3.14 + FastAPI + ChromaDB + HNSW + LangGraph + 60 FPS Canvas UI
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Find Python virtualenv or system python
if [ -d "./.venv" ]; then
    PYTHON="./.venv/bin/python3"
elif [ -d "$HOME/.venv" ]; then
    PYTHON="$HOME/.venv/bin/python3"
else
    PYTHON="python3"
fi

# Clean up any existing instances on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

echo "================================================================"
echo "🚀 Starting MY AI Python Vector Database & LangGraph Platform..."
echo "================================================================"

$PYTHON main.py
