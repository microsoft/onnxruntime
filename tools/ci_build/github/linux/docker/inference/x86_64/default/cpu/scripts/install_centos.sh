#!/bin/bash
set -e -x

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)
CPU_ARCH=`uname -m`
PACKAGES_TO_INSTALL="which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make libunwind  aria2  bzip2 bzip2-devel java-11-openjdk-devel graphviz devtoolset-10-binutils devtoolset-10-gcc devtoolset-10-gcc-c++ devtoolset-10-gcc-gfortran python3 python3-devel python3-pip"
if ! rpm -q --quiet epel-release ; then
	if [[ "$CPU_ARCH" = "x86_64" ]]; then
	  yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$os_major_version.noarch.rpm https://repo.ius.io/ius-release-el$os_major_version.rpm
	  PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL git224"
	else
	  yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$os_major_version.noarch.rpm
	  PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL git"
	fi  
fi

echo "installing for os major version : $os_major_version"
yum install -y centos-release-scl-rh
yum install -y $PACKAGES_TO_INSTALL

pip3 install --upgrade pip
localedef -i en_US -f UTF-8 en_US.UTF-8