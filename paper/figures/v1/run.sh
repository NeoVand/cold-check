#!/bin/sh
# Run a figure script with the paper's pinned plotting stack: ./run.sh fig1.py
cd "$(dirname "$0")"
exec uv run --offline --no-project --with numpy --with scipy --with matplotlib==3.10.7 --with fonttools --with pillow --with networkx python -I "$@"
