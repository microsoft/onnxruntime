#!/bin/bash
set -e -x
PYTHON_PREFIXES=("/opt/python/cp36-cp36m" "/opt/python/cp37-cp37m" "/opt/python/cp38-cp38" "/opt/python/cp39-cp39" "/opt/python/cp310-cp310")

for PREFIX in "${PYTHON_PREFIXES[@]}"; do
    PY_VER=$(${PREFIX}/bin/python -c "import sys; print('.'.join(str(v) for v in sys.version_info[:2]))")
    echo "Install packages for $PY_VER"
    if [ "$PY_VER" == "3.10" ]; then
        ${PREFIX}/bin/python -m pip install -r 310/requirements.txt
    else
        ${PREFIX}/bin/python -m pip install -r default/requirements.txt
    fi
done