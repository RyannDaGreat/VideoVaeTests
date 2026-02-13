#!/usr/bin/env bash
# Run manual tests. Thin wrapper around manual_tests/run_tests.py.
# Run with no arguments: bash run_manual_tests.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/manual_tests/run_tests.py" "$@"
