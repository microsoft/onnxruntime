#!/bin/bash
set -e
SCRIPT=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT`
TOP_SRC_DIR=`realpath $SCRIPT_DIR/../../../../`
mkdir -p $HOME/.cache/onnxruntime
if [ -z "$BUILD_ARTIFACTSTAGINGDIRECTORY" ]; then
  sudo mkdir -p /data/a
  BUILD_ARTIFACTSTAGINGDIRECTORY="/data/a"
fi

if [ -z "$BUILD_BINARIESDIRECTORY" ]; then
  sudo mkdir -p /data/b
  BUILD_BINARIESDIRECTORY="/data/b"
fi


for version in '23'; do
  docker_image=fedora$version
  cd $SCRIPT_DIR/docker
  docker build --pull -t $docker_image --build-arg OS_VERSION=$version -f Dockerfile.fedora .
  docker run --rm -e AZURESASKEY --volume "$HOME/.cache/onnxruntime:/root/.cache/onnxruntime" -v $BUILD_BINARIESDIRECTORY:/root/rpmbuild -v $BUILD_ARTIFACTSTAGINGDIRECTORY:/data/a -v $HOME/.ccache:/root/.ccache -v $TOP_SRC_DIR:/data/onnxruntime -w /data/b $docker_image /data/onnxruntime/tools/ci_build/github/linux/create_package_inside_docker.sh
done
