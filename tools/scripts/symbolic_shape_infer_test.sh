#!/bin/bash

set -ex

export build_dir=$1

# it's for manylinux image
export PATH=/opt/python/cp310-cp310/bin:$PATH

echo Run symbolic shape infer test
pushd $build_dir/Release/
python3 /build/Release/onnxruntime_test_python_symbolic_shape_infer.py
popd
