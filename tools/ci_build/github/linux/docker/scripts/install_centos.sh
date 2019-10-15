#!/bin/bash
set -e

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)


if ! rpm -q --quiet epel-release ; then
  yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$os_major_version.noarch.rpm
fi

echo "installing for os major version : $os_major_version"
if [ "$os_major_version" == "5" ]; then
  #Be careful, don't pull gcc into the base system, because we already have one in /opt/rh/devtoolset-2/root/usr/bin
  yum install -y redhat-lsb expat-devel libcurl-devel tar unzip curl zlib-devel make  python2-devel icu  rsync bzip2 git bzip2-devel
elif [ "$os_major_version" == "6" ]; then
  yum install -y centos-release-scl
  yum repolist
  yum install -y redhat-lsb-core expat-devel libcurl-devel tar unzip curl zlib-devel make libunwind icu aria2 rsync bzip2 git bzip2-devel
  yum upgrade -y
  yum install -y \
    ccache \
    devtoolset-7-binutils \
    devtoolset-7-gcc \
    devtoolset-7-gcc-c++ \
    devtoolset-7-gcc-gfortran 
  # The way to get python 3.6.8
  yum install -y https://centos6.iuscommunity.org/ius-release.rpm 
  yum --enablerepo=ius install -y python36u python36u-devel python36u-pip python36u-numpy python36u-setuptools python36u-wheel protobuf
  /usr/bin/python3.6 -m pip install --upgrade pip
else
  yum install -y redhat-lsb-core expat-devel libcurl-devel tar unzip curl zlib-devel make  python2-devel  libunwind icu aria2 rsync bzip2 git bzip2-devel
fi


#If the /opt/python folder exists, we assume this is the manylinux docker image
if [ "$os_major_version" != "6" ] && [ ! -d "/opt/python/cp35-cp35m"  ] 
then
  yum install -y ccache gcc gcc-c++ python3 python3-devel python3-pip python3-numpy python3-setuptools python3-wheel
fi
