#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
exec python "$script_dir/train.py" "$@"
