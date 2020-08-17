#!/bin/bash
apt-get -y update && apt-get -y install --no-install-recommends software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa -y && \
echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
apt-get update && apt-get install -y --allow-change-held-packages --allow-downgrades --no-install-recommends \
libcudnn7=7.6.5.32-1+cuda10.1 \
libcudnn7-dev=7.6.5.32-1+cuda10.1 \

