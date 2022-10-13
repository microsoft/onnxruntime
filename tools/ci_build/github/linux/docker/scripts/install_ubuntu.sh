#!/bin/bash
set -e
while getopts p:d: parameter_Option
do case "${parameter_Option}"
in
p) PYTHON_VER=${OPTARG};;
d) DEVICE_TYPE=${OPTARG};;
esac
done

PYTHON_VER=${PYTHON_VER:=3.8}
# Some Edge devices only have limited disk space, use this option to exclude some package
DEVICE_TYPE=${DEVICE_TYPE:=Normal}

DEBIAN_FRONTEND=noninteractive
echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

apt-get update && apt-get install -y software-properties-common lsb-release

OS_VERSION=$(lsb_release -r -s)

SYS_LONG_BIT=$(getconf LONG_BIT)

PACKAGE_LIST="autotools-dev \
	automake \
	build-essential \
	git apt-transport-https apt-utils \
	ca-certificates \
	pkg-config \
	wget \
	zlib1g \
	zlib1g-dev \
	libssl-dev \
	curl libcurl4-openssl-dev \
	autoconf \
	sudo \
	gfortran \
	python3-dev \
	language-pack-en \
	liblttng-ust0 \
	libcurl4 \
	libssl1.1 \
	libkrb5-3 \
	libtinfo-dev \
	libtinfo5 \
	libtool \
	openssh-server \
	aria2 \
	bzip2 \
	unzip \
	zip \
	rsync libunwind8 libpng-dev libexpat1-dev \
	python3-setuptools python3-numpy python3-wheel python python3-pip python3-pytest \
	openjdk-11-jdk \
	graphviz"


if [ $DEVICE_TYPE = "Normal" ]; then
    PACKAGE_LIST="$PACKAGE_LIST libedit-dev libxml2-dev python3-packaging"
fi

PACKAGE_LIST="$PACKAGE_LIST libicu66"

apt-get install -y --no-install-recommends $PACKAGE_LIST

locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

if [ "$OS_VERSION" = "20.04" ]; then
    if [ "$PYTHON_VER" != "3.8" ]; then
	    add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update
        apt-get install -y --no-install-recommends \
                python${PYTHON_VER} \
                python${PYTHON_VER}-dev
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VER} 1
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
        update-alternatives --set python3 /usr/bin/python${PYTHON_VER}
        #TODO: the old one(/usr/bin/pip3) should be uninstalled first. Because the one will be
        #put at /usr/local/. Then there will be two pips.
        /usr/bin/python${PYTHON_VER} -m pip install --upgrade --force-reinstall pip==19.0.3
    fi
else
    exit 1
fi

rm -rf /var/lib/apt/lists/*
