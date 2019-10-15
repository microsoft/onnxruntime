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
  yum upgrade -y
  yum install -y redhat-lsb-core expat-devel libcurl-devel tar unzip curl zlib-devel make  python2-devel  libunwind icu aria2 rsync bzip2 git bzip2-devel
  yum install -y \
    devtoolset-7-binutils \
    devtoolset-7-gcc \
    devtoolset-7-gcc-c++ \
    devtoolset-7-gcc-gfortran 
  yum install -y ccache rh-python36 rh-python36-devel rh-python36-pip rh-python36-numpy rh-python36-setuptools rh-python36-wheel
  /opt/rh/rh-python36/root/usr/bin/pip install --upgrade pip
else
  yum install -y redhat-lsb-core expat-devel libcurl-devel tar unzip curl zlib-devel make  python2-devel  libunwind icu aria2 rsync bzip2 git bzip2-devel
fi


#If the /opt/python folder exists, we assume this is the manylinux docker image
if [ "$os_major_version" != "6" ] && [ ! -d "/opt/python/cp35-cp35m"  ] 
then
  yum install -y ccache gcc gcc-c++ python3 python3-devel python3-pip python3-numpy python3-setuptools python3-wheel
fi
