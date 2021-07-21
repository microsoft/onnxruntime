#!/bin/bash
set -e
while getopts p:d: parameter_Option
do case "${parameter_Option}"
in
d) DEVICE_TYPE=${OPTARG};;
esac
done

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
	libicu66 \
	libtinfo-dev \
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
apt-get install -y --no-install-recommends $PACKAGE_LIST

locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

rm -rf /var/lib/apt/lists/*

if [ "$SYS_LONG_BIT" = "64" ]; then
    if [ "$DEVICE_TYPE" = "Normal" ]; then
	    if [ "$OS_VERSION" = "20.04" ]; then
	        #llvm 9.0 doesn't have a release for 20.04, but the binaries for 18.04 should work well.
            OS_VERSION="18.04"
	    fi
        aria2c -q -d /tmp -o llvm.tar.xz http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-${OS_VERSION}.tar.xz
	    tar --strip 1 -Jxf /tmp/llvm.tar.xz -C /usr
    fi
fi
