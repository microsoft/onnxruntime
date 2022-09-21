#!/bin/bash

# This script will run a baseline minimal ort build for android arm64-v8a ABI
# and write binary size data to a file

set -e
set -x

while getopts b:t:d parameter
do case "${parameter}"
in
b) BUILD_DIR="${OPTARG}";;
t) THRESHOLD_SIZE="${OPTARG}";;
d) INCLUDE_DEBUG_INFO=1;;
esac
done

if [[ -z "${BUILD_DIR}" ]]; then
    echo "Build directory must be specified with -b."
    exit 1
fi

CHECK_THRESHOLD_SIZE_ARGS=${THRESHOLD_SIZE:+"--threshold ${THRESHOLD_SIZE}"}
BUILD_WITH_DEBUG_INFO_ARGS=${INCLUDE_DEBUG_INFO:+"--cmake_extra_defines ADD_DEBUG_INFO_TO_MINIMAL_BUILD=ON"}

export PATH=/opt/python/cp37-cp37m/bin:$PATH

# Create an empty file to be used with build --include_ops_by_config, which will include no operators at all
mkdir -p ${BUILD_DIR}
echo -n > ${BUILD_DIR}/include_no_operators.config

# Run a baseline minimal build of ORT Android arm64-v8a
# Generate binary size as ${BUILD_DIR}/MinSizeRel/binary_size_data.txt
python3 /onnxruntime_src/tools/ci_build/build.py \
    --build_dir ${BUILD_DIR} --cmake_generator Ninja \
    --config MinSizeRel \
    --skip_submodule_sync \
    --parallel \
    --android \
    --android_sdk_path /android_home \
    --android_ndk_path /ndk_home \
    --android_abi=arm64-v8a \
    --android_api=29 \
    --minimal_build \
    --build_shared_lib \
    --build_java \
    --disable_ml_ops \
    --disable_exceptions \
    --include_ops_by_config ${BUILD_DIR}/include_no_operators.config \
    ${BUILD_WITH_DEBUG_INFO_ARGS}

python3 /onnxruntime_src/tools/ci_build/github/linux/ort_minimal/check_build_binary_size.py \
    ${CHECK_THRESHOLD_SIZE_ARGS} \
    ${BUILD_DIR}/MinSizeRel/libonnxruntime.so

echo "The content of binary_size_data.txt"
cat ${BUILD_DIR}/MinSizeRel/binary_size_data.txt
