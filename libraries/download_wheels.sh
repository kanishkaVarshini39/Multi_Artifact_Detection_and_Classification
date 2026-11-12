#!/usr/bin/env bash
# Fetch every wheel for an offline / archival rebuild of the environment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/wheels"
pip download -d "$HERE/wheels" -r "$HERE/requirements.txt"
pip download -d "$HERE/wheels" -r "$HERE/requirements-training.txt"
echo "wheels saved to $HERE/wheels"
