#!/usr/bin/env bash
# Regenerates the filamentcolors.xyz catalog bundled with AutoForge
# (src/autoforge/data/filamentcolors_catalog.json). Run this before each
# release: it crawls the whole filamentcolors.xyz API once (~25 requests,
# spaced out so the site isn't hammered) and keeps every swatch that has a
# measured TD. Installed webuis only fetch what was published after this
# snapshot, so a fresh snapshot keeps their startup checks tiny.
#
#   ./generate_filamentcolors_library.sh              # default output
#   ./generate_filamentcolors_library.sh --delay 5    # slower crawl (never
#                                                     # faster than 1 request/s)
#   PYTHON=/path/to/python ./generate_filamentcolors_library.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
OUTPUT="$SCRIPT_DIR/src/autoforge/data/filamentcolors_catalog.json"

# Run from the checkout even when autoforge isn't installed (or an older
# copy is): the module lives in src/.
PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m autoforge.Helper.filamentcolors_library --output "$OUTPUT" "$@"
