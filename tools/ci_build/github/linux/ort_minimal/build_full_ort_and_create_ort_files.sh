#!/bin/bash

# This script will run a full ORT build and use the python package built to generate ort format test files,
# which will be used in build_minimal_ort_and_run_tests.sh and nnapi_minimal_build_minimal_ort_and_run_tests.sh

set -e
set -x

# Validate the operator kernel registrations, as the ORT model uses hashes of the kernel registration details 
# to find kernels. If the hashes from the registration details are incorrect we will produce a model that will break
# when the registration is fixed in the future.
python3 /onnxruntime_src/tools/ci_build/op_registration_validator.py

# Run a full build of ORT.
# We need the ORT python package to generate the ORT format files and the required ops config files.
# We do not run tests since those are covered by other CIs
python3 /onnxruntime_src/tools/ci_build/build.py \
    --build_dir /build --cmake_generator Ninja \
    --config Debug \
    --skip_submodule_sync \
    --parallel \
    --build_wheel \
    --skip_tests \
    --enable_pybind

# Install the ORT python wheel
python3 -m pip install --user /build/Debug/dist/*

# Convert all the E2E ONNX models to ORT format 
python3 /onnxruntime_src/tools/python/convert_onnx_models_to_ort.py \
    /onnxruntime_src/onnxruntime/test/testdata/ort_minimal_e2e_test_data

# Create a config with just the required ops for ORT format models in testdata
# This is used by build_minimal_ort_and_run_tests.sh later in the linux-cpu-minimal-build-ci-pipeline CI
# and will include ops for the E2E models we just converted
python3 /onnxruntime_src/tools/python/create_reduced_build_config.py --format ORT \
    /onnxruntime_src/onnxruntime/test/testdata \
    /onnxruntime_src/onnxruntime/test/testdata/required_ops.ort_models.config

# Re-create testdata/required_ops_and_types.config.
# This is meaningful when nnapi_minimal_build_minimal_ort_and_run_tests.sh runs later in the
# linux-cpu-minimal-build-ci-pipeline CI, as recreating the configs checks that we are still creating
# a valid config. We have a checked in version of the file for use in other CIs where we don't do a full ORT
# build as part of the CI, such as the android-x86_64-crosscompile-ci-pipeline CI. 
python3 /onnxruntime_src/tools/python/create_reduced_build_config.py --format ORT --enable_type_reduction \
    /onnxruntime_src/onnxruntime/test/testdata \
    /onnxruntime_src/onnxruntime/test/testdata/required_ops_and_types.config

# Clear the build
rm -rf /build/Debug
