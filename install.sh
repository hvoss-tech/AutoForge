#!/usr/bin/env bash
# One-shot setup for AutoForge on Linux/macOS: installs `uv` (a fast Python
# package manager) if it's missing, installs all Python dependencies, and
# builds the web UI frontend. After this finishes, run ./run_webui.sh to
# start AutoForge.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== AutoForge installer ==="

if ! command -v uv &>/dev/null; then
    echo "[install] 'uv' not found - installing it now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv &>/dev/null; then
    echo "[install] ERROR: uv installed but isn't on PATH yet."
    echo "  Close and reopen your terminal and re-run ./install.sh, or see https://docs.astral.sh/uv/ for manual install steps."
    exit 1
fi

echo "[install] Installing Python dependencies with uv..."
uv sync

# `uv sync` installs PyPI's default PyTorch build: CUDA on Linux/Windows, MPS
# (Apple Metal) on macOS. Two kinds of machine need a different build:
#
# * Pre-Turing NVIDIA GPUs (compute capability < 7.5) have no kernels in the
#   default CUDA build. torch.cuda.is_available() still returns True on these
#   cards, so the failure only surfaces at the first kernel launch as "no
#   kernel image is available for execution on the device". The CUDA 12.6
#   build is the last one shipping Maxwell/Pascal/Volta kernels.
# * AMD GPUs need the ROCm build; with the CUDA one AutoForge silently runs on
#   the CPU.
#
# AUTOFORGE_TORCH_INDEX=<PyTorch wheel index URL> skips the detection and
# installs from that index instead (e.g. https://download.pytorch.org/whl/cpu).
#
# `uv run` and `uv sync` re-sync the venv to the lockfile and would undo the
# swap, so the index is recorded in a marker file that run_webui.sh and the
# updater read.
TORCH_INDEX_MARKER=".venv/.autoforge-torch-index"
LEGACY_CU126_MARKER=".venv/.autoforge-torch-cu126"
rm -f "$TORCH_INDEX_MARKER" "$LEGACY_CU126_MARKER"

TORCH_INDEX="${AUTOFORGE_TORCH_INDEX:-}"
NVIDIA_GPU=false
if [ -z "$TORCH_INDEX" ] && command -v nvidia-smi &>/dev/null; then
    # Lowest compute capability across all GPUs, as major*10+minor (6.1 -> 61)
    # so the comparison is integer-only. Non-numeric output (old drivers that
    # lack the compute_cap field, "[N/A]", error text) is ignored.
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' \r' \
        | awk -F. '/^[0-9]+\.[0-9]+$/ { v = $1 * 10 + $2; if (m == "" || v < m) m = v } END { if (m != "") print m }')"
    if [ -n "$CC" ]; then
        NVIDIA_GPU=true
        if [ "$CC" -lt 75 ]; then
            echo "[install] Detected GPU with compute capability $((CC / 10)).$((CC % 10)) (pre-Turing)."
            TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        fi
    fi
fi
# /dev/kfd is the ROCm compute interface the amdgpu driver exposes.
if [ -z "$TORCH_INDEX" ] && [ "$NVIDIA_GPU" = false ] && [ "$(uname -s)" = "Linux" ] && [ -e /dev/kfd ]; then
    echo "[install] Detected an AMD GPU (ROCm)."
    TORCH_INDEX="https://download.pytorch.org/whl/rocm7.1"
fi

if [ -n "$TORCH_INDEX" ]; then
    echo "[install] Reinstalling PyTorch from $TORCH_INDEX ..."
    # Version floors mirror pyproject.toml; the newest build the index carries
    # is installed, so this keeps working if PyPI's default moves on.
    if uv pip install --reinstall-package torch --reinstall-package torchvision \
        "torch>=2.9.1" "torchvision>=0.21.0" \
        --index-url "$TORCH_INDEX"; then
        mkdir -p "$(dirname "$TORCH_INDEX_MARKER")" && echo "$TORCH_INDEX" > "$TORCH_INDEX_MARKER"
    else
        echo "[install] ERROR: could not install PyTorch from $TORCH_INDEX."
        echo "  See the 'GPU support' section of the README."
        exit 1
    fi
fi

if command -v npm &>/dev/null; then
    echo "[install] Building the web UI frontend (this can take a minute)..."
    (cd webui/frontend && npm install && npm run build)
else
    echo "[install] WARNING: npm not found - skipping the web UI frontend build."
    echo "  Install Node.js (https://nodejs.org/) if you want to use the web UI, then re-run ./install.sh."
fi

echo
echo "=== Install complete ==="
echo "Start AutoForge with:   ./run_webui.sh"
echo "Check for updates with: ./update.sh"
echo
echo "Note: the web UI sends anonymous usage telemetry (PostHog) by default."
echo "  Disable it with ./run_webui.sh --no-telemetry, or permanently via"
echo "  AUTOFORGE_WEBUI_TELEMETRY_ENABLED=false. See README.md for details."
