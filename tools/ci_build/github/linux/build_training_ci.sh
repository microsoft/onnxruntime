#!/bin/bash
set -e -x
python3.12 -m pip install -r /onnxruntime_src/tools/ci_build/github/linux/python/requirements.txt
python3.12 /onnxruntime_src/tools/ci_build/build.py --build_dir /build --config Release --enable_training --skip_submodule_sync --parallel
