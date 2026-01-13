#!/bin/bash
# Run a Python script or command from the fano root
cd "$(dirname "$0")/.."
FANO_ROOT=$(pwd)

if [ $# -eq 0 ]; then
    echo "Usage: run_py.sh <script.py> [args...]"
    echo "   or: run_py.sh -c 'python code'"
    exit 1
fi

cd "$FANO_ROOT"
python "$@"
