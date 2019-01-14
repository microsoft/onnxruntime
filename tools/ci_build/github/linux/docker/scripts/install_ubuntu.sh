#!/bin/bash
set -e
while getopts p: parameter_Option
do case "${parameter_Option}"
in
p) PYTHON_VER=${OPTARG};;
esac
done

DEBIAN_FRONTEND=noninteractive
mkdir -p /tmp/dotnet
aria2c -q -d /tmp/dotnet https://packages.microsoft.com/config/ubuntu/16.04/packages-microsoft-prod.deb
dpkg -i /tmp/dotnet/packages-microsoft-prod.deb
apt-get update && apt-get install -y software-properties-common \
        apt-transport-https
add-apt-repository ppa:deadsnakes/ppa
apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        git ca-certificates \
        ca-certificates \
        pkg-config \
        wget \
        zlib1g \
        zlib1g-dev \
        libssl-dev \
        curl \
        autoconf \
        sudo \
        gfortran \
        python3-dev \
        language-pack-en \
        libopenblas-dev \
        liblttng-ust0 \
        libcurl3 \
        libssl1.0.0 \
        libkrb5-3 \
        libicu55 \
        aria2 \
        bzip2 \
        unzip \
        zip \
        rsync libunwind8 libpng16-dev \
        python3-setuptools python3-numpy python3-wheel python python3-pip python3-pytest \
        dotnet-sdk-2.2

locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

if [ $PYTHON_VER != "3.5" ]; then
    apt-get install -y --no-install-recommends \
            python${PYTHON_VER} \
            python${PYTHON_VER}-dev
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VER} 1
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 2
    update-alternatives --set python3 /usr/bin/python${PYTHON_VER}
    /usr/bin/python${PYTHON_VER} -m pip install --upgrade --force-reinstall numpy
fi

rm -rf /var/lib/apt/lists/*

mkdir -p /tmp/azcopy
aria2c -q -d /tmp/azcopy -o azcopy.tar.gz https://aka.ms/downloadazcopylinux64
tar -xf /tmp/azcopy/azcopy.tar.gz -C /tmp/azcopy
/tmp/azcopy/install.sh
