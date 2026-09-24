#!/usr/bin/env bash
# Measure every label on every figure against the width the manuscript places it
# at, and refuse to stay quiet about one that is too small or that sits on top of
# another.
#
#   ./checkfigs.sh                    every figure in figures/
#   ./checkfigs.sh figures/fig1_map.pdf   one of them
#
# figcheck needs pdfplumber, which wants a Pillow newer than several other
# packages accept, so it belongs in an environment of its own:
#
#     python3 -m venv .venv-figcheck
#     .venv-figcheck/bin/pip install -q pdfplumber
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
if [ -x .venv-figcheck/bin/python3 ] && .venv-figcheck/bin/python3 -c 'import pdfplumber' 2>/dev/null; then
    PY=.venv-figcheck/bin/python3
fi

if ! "$PY" -c 'import pdfplumber' 2>/dev/null; then
    echo "figcheck needs pdfplumber and no environment here has it." >&2
    echo "Do not install it into your main environment; it will pull a Pillow" >&2
    echo "version other packages may not accept. Make it its own:" >&2
    echo >&2
    echo "    python3 -m venv .venv-figcheck" >&2
    echo "    .venv-figcheck/bin/pip install -q pdfplumber" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    exec "$PY" src/figcheck.py "$@"
fi
if ! compgen -G 'figures/*.pdf' > /dev/null; then
    echo "no figures in figures/ yet; run the src/*.py scripts first (see README)" >&2
    exit 1
fi
exec "$PY" src/figcheck.py figures/*.pdf
