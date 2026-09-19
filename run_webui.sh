#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${WEBUI_HOST:-0.0.0.0}"
PORT="${WEBUI_PORT:-8000}"
BUILD_FRONTEND="${BUILD_FRONTEND:-true}"

# ---------------------------------------------------------------------------
# Build the frontend if the dist folder is missing (or forced)
# ---------------------------------------------------------------------------
FRONTEND_DIST="$SCRIPT_DIR/webui/frontend/dist"

should_build=false
if [ "$BUILD_FRONTEND" = "true" ]; then
    should_build=true
elif [ "$BUILD_FRONTEND" = "auto" ] && [ ! -d "$FRONTEND_DIST" ]; then
    should_build=true
fi

if [ "$should_build" = true ]; then
    echo "[webui] Building frontend..."
    if ! command -v npm &>/dev/null; then
        echo "[webui] ERROR: npm not found. Install Node.js or set BUILD_FRONTEND=false."
        exit 1
    fi
    cd "$SCRIPT_DIR/webui/frontend"
    npm install
    npm run build
    cd "$SCRIPT_DIR"
    echo "[webui] Frontend built."
fi

# ---------------------------------------------------------------------------
# Start the FastAPI server
# ---------------------------------------------------------------------------
echo "[webui] Starting server on http://${HOST}:${PORT}"

if [ "${NO_BROWSER:-false}" != "true" ]; then
    BROWSER_URL="http://localhost:${PORT}"
    (
        sleep 2
        if command -v xdg-open &>/dev/null; then
            xdg-open "$BROWSER_URL" &>/dev/null
        elif command -v open &>/dev/null; then
            open "$BROWSER_URL" &>/dev/null
        fi
    ) &
fi

# install.sh installs a pre-Turing-compatible PyTorch build on old NVIDIA GPUs;
# `uv run` would sync it back to the default build, so skip syncing in that case.
if [ -f "$SCRIPT_DIR/.venv/.autoforge-torch-cu126" ]; then
    export UV_NO_SYNC=1
fi

exec uv run uvicorn autoforge.webui.server:app \
    --host "$HOST" \
    --port "$PORT" \
    "${EXTRA_UVICORN_ARGS[@]}"
