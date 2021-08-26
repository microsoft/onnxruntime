#!/bin/bash
set -e -x
for PREFIX in $(find /opt/python/ -mindepth 1 -maxdepth 1 \( -name 'cp*' -o -name 'pp*' \)); do
    PY_VER=$(${PREFIX}/bin/python -c "import sys; print('.'.join(str(v) for v in sys.version_info[:2]))")
    echo "Install packages for $PY_VER"
    if [ "$PY_VER" == "3.10" ]; then
        ${PREFIX}/bin/python -m pip install -r 310/requirements.txt
    else
        ${PREFIX}/bin/python -m pip install -r default/requirements.txt
    fi
done