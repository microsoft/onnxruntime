#!/bin/bash


apt-get update &&\
 apt-get install -y sudo git bash

cd  /code

ONNXRUNTIME_SERVER_BRANCH=$1
COMMIT=$2

# Dependencies: cmake
wget --quiet https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz
tar zxf cmake-3.14.3-Linux-x86_64.tar.gz

export PATH=/code/cmake-3.14.3-Linux-x86_64/bin:${PATH}

GIT_TOKEN=[CHANGE THIS]
ONNXRUNTIME_REPO=https://${GIT_TOKEN}@aiinfra.visualstudio.com/Lotus/_git/onnxruntime
git clone --single-branch --branch ${ONNXRUNTIME_SERVER_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime

# Prepare onnxruntime repository & build onnxruntime
cd onnxruntime

if [ -z "${COMMIT}" ]
then
  echo "use latest commit of current branch"
else
  git checkout ${COMMIT}
fi

# get commit id
COMMITID=$(git rev-parse HEAD | cut -c1-8)
echo "Branch: "$ONNXRUNTIME_SERVER_BRANCH", commit: "$COMMITID

bash ./build.sh --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_cuda --config Release --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER)

cd /tmp
BINARY_FOLDER=binary_$COMMITID
rm -rf  $BINARY_FOLDER
mkdir $BINARY_FOLDER
cp -rf /code/onnxruntime/build/Linux/Release/* $BINARY_FOLDER

tar czvf $BINARY_FOLDER.tar.gz $BINARY_FOLDER
curl -sL https://aka.ms/InstallAzureCLIDeb | bash
az storage blob upload --container-name philly --account-name onnxtraining --name $BINARY_FOLDER.tar.gz --account-key [CHANGE THIS] --file $BINARY_FOLDER.tar.gz

echo "Uploaded your build binary to https://onnxtraining.blob.core.windows.net/philly/"${BINARY_FOLDER}".tar.gz, be noted, it is accessible to all public"