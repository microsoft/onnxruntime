#!/bin/bash
set -e
while getopts p:d: parameter_Option
do case "${parameter_Option}"
in
p) PYTHON_VER=${OPTARG};;
d) DEVICE_TYPE=${OPTARG};;
*) echo "Usage: $0 -p PYTHON_VER -d DEVICE_TYPE";;
esac
done

PYTHON_VER=${PYTHON_VER:=3.8}
# Some Edge devices only have limited disk space, use this option to exclude some package
DEVICE_TYPE=${DEVICE_TYPE:=Normal}

# shellcheck disable=SC2034
DEBIAN_FRONTEND=noninteractive
echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

apt-get update && apt-get install -y software-properties-common lsb-release

OS_VERSION=$(lsb_release -r -s)

PACKAGE_LIST=(
    "apt-transport-https"
    "apt-utils"
    "aria2"
    "autoconf"
    "automake"
    "autotools-dev"
    "build-essential"
    "bzip2"
    "ca-certificates"
    "curl"
    "gfortran"
    "git"
    "graphviz"
    "language-pack-en"
    "libcurl4"
    "libcurl4-openssl-dev"
    "libexpat1-dev"
    "libkrb5-3"
    "liblttng-ust-dev"
    "libpng-dev"
    "libssl-dev"
    "libtinfo-dev"
    "libtinfo5"
    "libtool"
    "libunwind8"
    "openjdk-17-jdk"
    "openssh-server"
    "pkg-config"
    "python3-dev"
    "python3-distutils"
    "python3-numpy"
    "python3-pip"
    "python3-pytest"
    "python3-setuptools"
    "python3-wheel"
    "rsync"
    "sudo"
    "unzip"
    "wget"
    "zip"
    "zlib1g"
    "zlib1g-dev"
)
if [ "$DEVICE_TYPE" = "Normal" ]; then
    PACKAGE_LIST+=("libedit-dev" "libxml2-dev" "python3-packaging")
fi

PACKAGE_LIST+=("libicu-dev")

apt-get install -y --no-install-recommends "${PACKAGE_LIST[@]}"

locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

if [ "$OS_VERSION" = "20.04" ]; then
  # The defaul version of python is 3.8
    major=$(echo "$PYTHON_VER" | cut -d. -f1)
    minor=$(echo "$PYTHON_VER" | cut -d. -f2)
    if [ "$major" -lt 3 ] || [ "$major" -eq 3 ] && [ "$minor" -lt 8 ]; then
      PYTHON_VER="3.8"
    fi
    if [ "$PYTHON_VER" != "3.8" ]; then
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update
        apt-get install -y --no-install-recommends \
                python"${PYTHON_VER}" \
                python"${PYTHON_VER}-"dev
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python"${PYTHON_VER}" 1
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
        update-alternatives --set python3 /usr/bin/python"${PYTHON_VER}"
        #TODO: the old one(/usr/bin/pip3) should be uninstalled first. Because the one will be
        #put at /usr/local/. Then there will be two pips.
        /usr/bin/python"${PYTHON_VER}" -m pip install --upgrade --force-reinstall pip==19.0.3
    fi
elif [ "$OS_VERSION" = "22.04" ] ; then
  # The defaul version of python is 3.10
    major=$(echo "$PYTHON_VER" | cut -d. -f1)
    minor=$(echo "$PYTHON_VER" | cut -d. -f2)
    if [ "$major" -lt 3 ] || [ "$major" -eq 3 ] && [ "$minor" -lt 10 ]; then
      PYTHON_VER="3.10"
    fi
    if [ "$PYTHON_VER" != "3.10" ]; then
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update
        apt-get install -y --no-install-recommends \
                python"${PYTHON_VER}" \
                python"${PYTHON_VER}"-dev
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python"${PYTHON_VER}" 1
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
        update-alternatives --set python3 /usr/bin/python"${PYTHON_VER}"
    fi
else
    exit 1
fi

rm -rf /var/lib/apt/lists/*
