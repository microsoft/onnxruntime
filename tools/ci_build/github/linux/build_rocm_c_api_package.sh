#!/bin/bash

set -e -u -x

usage() { echo "Usage: $0 -S <source dir> -B <binary dir> -V <rocm version> [-H <rocm home>] " 1>&2; exit 1; }

ROCM_HOME=/opt/rocm

while getopts S:B:V:H:I:P: parameter_Option; do
  case "${parameter_Option}" in
    S) SOURCE_DIR=${OPTARG};;
    B) BINARY_DIR=${OPTARG};;
    V) ROCM_VERSION=${OPTARG};;
    H) ROCM_HOME=${OPTARG};;
    I) IMAGE=${OPTARG};;
    P) PYTHON_BIN=${OPTARG};;
    *) usage ;;
  esac
done

EXIT_CODE=1

docker run --rm \
  --security-opt seccomp=unconfined \
  --shm-size=1024m \
  --user $UID:$(id -g $USER) \
  -e CFLAGS="-Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong -fstack-clash-protection -fcf-protection -O3 -Wl,--strip-all" \
  -e CXXFLAGS="-Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong -fstack-clash-protection -fcf-protection -O3 -Wl,--strip-all" \
  -e NIGHTLY_BUILD \
  --volume $SOURCE_DIR:/onnxruntime_src \
  --volume $BINARY_DIR:/build \
  --volume /data/models:/build/models:ro \
  --volume /data/onnx:/data/onnx:ro \
  --workdir /onnxruntime_src \
  $IMAGE \
  ${PYTHON_BIN:-python} /onnxruntime_src/tools/ci_build/build.py \
    --config Release \
    --build_dir /build \
    --parallel \
    --use_rocm --rocm_version=$ROCM_VERSION --rocm_home $ROCM_HOME --nccl_home $ROCM_HOME \
    --build_shared_lib \
    --skip_submodule_sync \
    --skip_tests --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER


EXIT_CODE=$?

set -e
exit $EXIT_CODE
