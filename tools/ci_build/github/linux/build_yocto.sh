#!/bin/bash
set -e -o -x
SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
TARGET_FOLDER="/datadrive/ARM"
SOURCE_ROOT=$(realpath $SCRIPT_DIR/../../../../)
YOCTO_VERSION="4.19"

while getopts f:y: parameter_Option
do case "${parameter_Option}"
in
f) TARGET_FOLDER=${OPTARG};;
y) YOCTO_VERSION=${OPTARG};;
esac
done

YOCTO_IMAGE="arm-yocto"
IMX_BRANCH="imx-linux-warrior"
IMX_MANIFEST="imx-4.19.35-1.1.0.xml"

if [ $YOCTO_VERSION = "4.14" ]; then
    IMX_BRANCH="imx-linux-sumo"
    IMX_MANIFEST="imx-4.14.98-2.0.0_machinelearning.xml"
fi

cd $SCRIPT_DIR/docker
docker build -t $YOCTO_IMAGE -f Dockerfile.arm_yocto .

if [ ! -f $TARGET_FOLDER/bin/repo ]; then
    mkdir $TARGET_FOLDER/bin
    curl https://storage.googleapis.com/git-repo-downloads/repo > $TARGET_FOLDER/bin/repo
    chmod a+x $TARGET_FOLDER/bin/repo
fi

if [ ! -d $TARGET_FOLDER/fsl-arm-yocto-bsp ]; then
    mkdir $TARGET_FOLDER/fsl-arm-yocto-bsp
    cd $TARGET_FOLDER/fsl-arm-yocto-bsp
    $TARGET_FOLDER/bin/repo init -u https://source.codeaurora.org/external/imx/imx-manifest -b $IMX_BRANCH -m $IMX_MANIFEST
    $TARGET_FOLDER/bin/repo sync
fi

YOCTO_CONTAINER="arm_yocto"
docker rm -f $YOCTO_CONTAINER || true
docker run --name $YOCTO_CONTAINER --volume $TARGET_FOLDER/fsl-arm-yocto-bsp:/fsl-arm-yocto-bsp --volume $SOURCE_ROOT:/onnxruntime_src $YOCTO_IMAGE /bin/bash /onnxruntime_src/tools/ci_build/github/linux/yocto_build_toolchain.sh -y $YOCTO_VERSION &

wait $!

EXIT_CODE=$?

set -e
exit $EXIT_CODE