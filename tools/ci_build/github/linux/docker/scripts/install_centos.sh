#!/bin/bash
set -e

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)


yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$os_major_version.noarch.rpm

if [ "$os_major_version" == "5" ]; then
 yum install -y redhat-lsb expat-devel libcurl-devel rpmdevtools tar unzip curl zlib-devel make  python2-devel icu  rsync bzip2 git bzip2-devel
else
 yum install -y redhat-lsb-core expat-devel libcurl-devel rpmdevtools tar unzip curl zlib-devel make  python2-devel  libunwind icu aria2 rsync bzip2 git bzip2-devel
fi


#If the /opt/python folder exists, we assume this is the manylinux docker image
if [ ! -d "/opt/python/cp35-cp35m"  ] 
then
yum install -y ccache gcc gcc-c++ python3-devel python3-pip python3-numpy python3-setuptools python3-wheel
fi
