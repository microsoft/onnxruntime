#!/bin/bash
set -e -x
BUILD_CONFIG="Release"

while getopts "i:d:x:c:" parameter_Option
do case "${parameter_Option}"
in
i) DOCKER_IMAGE=${OPTARG};;
d) DEVICE=${OPTARG};;
x) BUILD_EXTR_PAR=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
*) echo "Usage: $0 -i <docker_image> -d <GPU|CPU> [-x <extra_build_arg>] [-c <build_config>]"
   exit 1;;
esac
done

mkdir -p "${HOME}/.onnx"
DOCKER_SCRIPT_OPTIONS="-d ${DEVICE} -c ${BUILD_CONFIG}"

if [ "${BUILD_EXTR_PAR}" != "" ] ; then
    DOCKER_SCRIPT_OPTIONS+=" -x ${BUILD_EXTR_PAR}"
fi

docker run --rm \
    --volume /data/onnx:/data/onnx:ro \
    --volume "${BUILD_SOURCESDIRECTORY}:/onnxruntime_src" \
    --volume "${BUILD_BINARIESDIRECTORY}:/build" \
    --volume /data/models:/build/models:ro \
    --volume "${HOME}/.onnx:/home/onnxruntimedev/.onnx" \
    -w /onnxruntime_src \
    -e NIGHTLY_BUILD \
    -e BUILD_BUILDNUMBER \
    -e ORT_DISABLE_PYTHON_PACKAGE_LOCAL_VERSION \
    -e DEFAULT_TRAINING_PACKAGE_DEVICE \
    $ADDITIONAL_DOCKER_PARAMETER \
    $DOCKER_IMAGE tools/ci_build/github/linux/build_linux_python_package.sh $DOCKER_SCRIPT_OPTIONS

sudo rm -rf "${BUILD_BINARIESDIRECTORY}/${BUILD_CONFIG}/onnxruntime" "${BUILD_BINARIESDIRECTORY}/${BUILD_CONFIG}/pybind11" \
    "${BUILD_BINARIESDIRECTORY}/${BUILD_CONFIG}/models" "${BUILD_BINARIESDIRECTORY}/${BUILD_CONFIG}/_deps" \
    "${BUILD_BINARIESDIRECTORY}/${BUILD_CONFIG}/CMakeFiles"
cd "${BUILD_BINARIESDIRECTORY}/${BUILD_CONFIG}"
find -executable -type f > "${BUILD_BINARIESDIRECTORY}/${BUILD_CONFIG}/perms.txt"
