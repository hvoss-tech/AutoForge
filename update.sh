#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Don't let `uv run` swap a ROCm / CUDA 12.6 PyTorch that install.sh put in
# back to the default build; update.py reinstalls it after its own sync.
if [ -f .venv/.autoforge-torch-index ] || [ -f .venv/.autoforge-torch-cu126 ]; then
    export UV_NO_SYNC=1
fi
uv run python update.py "$@"
