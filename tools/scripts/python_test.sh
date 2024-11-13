#!/bin/bash

set -ex

export src_dir=$1
export build_dir=$2
export config=$3

# it's for manylinux image
export PATH=/opt/python/cp310-cp310/bin:$PATH

echo Install Python Deps
cp $src_dir/tools/ci_build/github/linux/docker/scripts/manylinux/requirements.txt $build_dir/requirements.txt

python3 -m pip install -r $build_dir/requirements.txt
python3 -m pip list | grep onnx

echo Install $config python package
rm -rf $build_dir/$config/onnxruntime $build_dir/$config/pybind11
python3 -m pip install $build_dir/$config/dist/*.whl

echo Run $config unit tests
cd $build_dir/$config/
python3 $src_dir/tools/ci_build/build.py --build_dir $build_dir --cmake_generator Ninja --config $config --test --skip_submodule_sync --build_shared_lib --parallel --use_binskim_compliant_compile_flags  --build_wheel --enable_onnx_tests --enable_transformers_tool_test
