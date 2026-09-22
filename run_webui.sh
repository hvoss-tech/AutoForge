#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${WEBUI_HOST:-0.0.0.0}"
PORT="${WEBUI_PORT:-8000}"
BUILD_FRONTEND="${BUILD_FRONTEND:-auto}"

# ---------------------------------------------------------------------------
# Build the frontend, but only when it's actually out of date. BUILD_FRONTEND
# used to default to "true", which ran a full `npm install && npm run build`
# (tens of seconds) on every single start even when nothing in the frontend
# had changed since the last build. "auto" (now the default) rebuilds only
# when dist/ is missing or a source/config file is newer than the last
# successful build's stamp; BUILD_FRONTEND=true still forces a rebuild every
# time, and BUILD_FRONTEND=false always skips it.
# ---------------------------------------------------------------------------
FRONTEND_DIR="$SCRIPT_DIR/webui/frontend"
FRONTEND_DIST="$FRONTEND_DIR/dist"
BUILD_STAMP="$FRONTEND_DIST/.build-stamp"

should_build=false
if [ "$BUILD_FRONTEND" = "true" ]; then
    should_build=true
elif [ "$BUILD_FRONTEND" = "false" ]; then
    should_build=false
else
    if [ ! -f "$BUILD_STAMP" ]; then
        should_build=true
    elif find "$FRONTEND_DIR/src" "$FRONTEND_DIR/package.json" "$FRONTEND_DIR/package-lock.json" \
              "$FRONTEND_DIR/vite.config.ts" "$FRONTEND_DIR/index.html" "$FRONTEND_DIR/tailwind.config.js" \
              "$FRONTEND_DIR/postcss.config.js" "$FRONTEND_DIR/tsconfig.json" \
              -newer "$BUILD_STAMP" 2>/dev/null | grep -q .; then
        should_build=true
    fi
fi

if [ "$should_build" = true ]; then
    echo "[webui] Building frontend..."
    if ! command -v npm &>/dev/null; then
        echo "[webui] ERROR: npm not found. Install Node.js or set BUILD_FRONTEND=false."
        exit 1
    fi
    cd "$FRONTEND_DIR"
    npm install
    npm run build
    mkdir -p "$FRONTEND_DIST"
    touch "$BUILD_STAMP"
    cd "$SCRIPT_DIR"
    echo "[webui] Frontend built."
else
    echo "[webui] Frontend up to date, skipping build (set BUILD_FRONTEND=true to force)."
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
