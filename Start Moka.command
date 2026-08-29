#!/bin/bash
set -u
cd "$(dirname "$0")" || exit 1
PY=""
for candidate in python3.12 python3.13 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11–3.13 is required. Install Python 3.12, then run this file again."
  read -r -p "Press Enter to close."
  exit 1
fi
"$PY" tools/launch.py
RESULT=$?
if [ "$RESULT" -ne 0 ]; then read -r -p "Startup failed. Read the error above; press Enter to close."; fi
exit "$RESULT"
