#!/usr/bin/env bash
set -euo pipefail
umask 077
REBOUND_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REBOUND_PYTHON="${REBOUND_PYTHON:-python3}"
REBOUND_ENV="${REBOUND_ENV:-$REBOUND_ROOT/../.rebound-training-venv}"
export PYTHONDONTWRITEBYTECODE=1
cd "$REBOUND_ROOT"
"$REBOUND_PYTHON" -c 'import sys; assert sys.version_info[:2] in [(3, 10), (3, 11)], "Use Python 3.10 or 3.11"'
if [[ ! -x "$REBOUND_ENV/bin/python" ]]; then
  "$REBOUND_PYTHON" -m venv "$REBOUND_ENV"
fi
"$REBOUND_ENV/bin/python" -m pip install --only-binary=:all: --no-cache-dir -r "$REBOUND_ROOT/requirements.txt"
exec "$REBOUND_ENV/bin/python" -B "$REBOUND_ROOT/train.py" --reproduction false "$@"
