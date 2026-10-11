#!/usr/bin/env bash
set -euo pipefail
umask 077
REBOUND_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REBOUND_PYTHON="${REBOUND_PYTHON:-python3}"
REBOUND_ENV="${REBOUND_ENV:-$REBOUND_ROOT/../.rebound-venv}"
export PYTHONDONTWRITEBYTECODE=1
cd "$REBOUND_ROOT"
"$REBOUND_PYTHON" -c 'import sys; assert sys.version_info[:2] in [(3, 10), (3, 11)], "Use Python 3.10 or 3.11"'
if [[ ! -x "$REBOUND_ENV/bin/python" ]]; then
  "$REBOUND_PYTHON" -m venv "$REBOUND_ENV"
fi
if ! "$REBOUND_ENV/bin/python" -c 'import importlib.metadata as m; assert all(m.version(k)==v for k,v in {"numpy":"1.26.4","scipy":"1.13.1","scikit-learn":"1.7.2","joblib":"1.6.0","threadpoolctl":"3.6.0","cloudpickle":"3.1.2"}.items())' 2>/dev/null; then
  "$REBOUND_ENV/bin/python" -m pip install --only-binary=:all: --no-cache-dir -r "$REBOUND_ROOT/requirements.txt"
fi
exec "$REBOUND_ENV/bin/python" -B "$REBOUND_ROOT/reproduce.py" "$@"
