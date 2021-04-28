#!/bin/bash

# This script will run a full ORT build and use the python package built to generate ort format test files,
# which will be used in build_minimal_ort_and_run_tests.sh and nnapi_minimal_build_minimal_ort_and_run_tests.sh

set -e
set -x
export PATH=/opt/python/cp37-cp37m/bin:$PATH

# Validate the operator kernel registrations, as the ORT model uses hashes of the kernel registration details
# to find kernels. If the hashes from the registration details are incorrect we will produce a model that will break
# when the registration is fixed in the future.
python3 /onnxruntime_src/tools/ci_build/op_registration_validator.py

# Run a full build of ORT.
# We need the ORT python package to generate the ORT format files and the required ops config files.
# We do not run tests in this command since those are covered by other CIs.
python3 /onnxruntime_src/tools/ci_build/build.py \
    --build_dir /build --cmake_generator Ninja \
    --config Debug \
    --skip_submodule_sync \
    --parallel \
    --build_wheel \
    --skip_tests \
    --enable_training_ops \
    --enable_pybind --cmake_extra_defines PYTHON_INCLUDE_DIR=/opt/python/cp37-cp37m/include/python3.7m PYTHON_LIBRARY=/usr/lib64/librt.so

# Run kernel def hash verification test
pushd /build/Debug
ORT_TEST_STRICT_KERNEL_DEF_HASH_CHECK=1 ./onnxruntime_test_all --gtest_filter="KernelDefHashTest.ExpectedCpuKernelDefHashes"
popd

# Install the ORT python wheel
python3 -m pip install --user /build/Debug/dist/*

# Convert all the E2E ONNX models to ORT format
python3 /onnxruntime_src/tools/python/convert_onnx_models_to_ort.py \
    /onnxruntime_src/onnxruntime/test/testdata/ort_minimal_e2e_test_data

# Create configs with just the required ops for ORT format models in testdata
# These are used by build_minimal_ort_and_run_tests.sh later in the linux-cpu-minimal-build-ci-pipeline CI
# and will include ops for the E2E models we just converted

# Config without type reduction
python3 /onnxruntime_src/tools/python/create_reduced_build_config.py --format ORT \
    /onnxruntime_src/onnxruntime/test/testdata \
    /home/onnxruntimedev/.test_data/required_ops.ort_models.config

# Config with type reduction
python3 /onnxruntime_src/tools/python/create_reduced_build_config.py --format ORT --enable_type_reduction \
    /onnxruntime_src/onnxruntime/test/testdata \
    /home/onnxruntimedev/.test_data/required_ops_and_types.ort_models.config

# Test that we can convert an ONNX model with custom ops to ORT format
mkdir /home/onnxruntimedev/.test_data/custom_ops_model
cp /onnxruntime_src/onnxruntime/test/testdata/custom_op_library/*.onnx /home/onnxruntimedev/.test_data/custom_ops_model/
python3 /onnxruntime_src/tools/python/convert_onnx_models_to_ort.py \
    --custom_op_library /build/Debug/libcustom_op_library.so \
    /home/onnxruntimedev/.test_data/custom_ops_model
rm -rf /home/onnxruntimedev/.test_data/custom_ops_model

# Clear the build
rm -rf /build/Debug
