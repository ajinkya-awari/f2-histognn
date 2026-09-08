#!/usr/bin/env sh
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
python -m compileall -q data models explanations training tests
python -m pytest -q
