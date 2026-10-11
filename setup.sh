#!/usr/bin/env bash
set -euo pipefail

REBOUND_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REBOUND_PYTHON="${REBOUND_PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1

"$REBOUND_PYTHON" -c 'import sys; assert sys.version_info[:2] in [(3, 10), (3, 11)], "Use Python 3.10 or 3.11"'
exec "$REBOUND_PYTHON" -m pip install --only-binary=:all: --no-cache-dir -r "$REBOUND_ROOT/requirements.txt"
