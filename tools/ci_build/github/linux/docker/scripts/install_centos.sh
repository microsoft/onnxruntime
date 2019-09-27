#!/bin/bash
set -e

yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm https://centos7.iuscommunity.org/ius-release.rpm

yum remove --tolerant cmake git
yum install -y cmake3 git2u-all
ln -s /usr/bin/cmake3 /usr/bin/cmake
ln -s /usr/bin/ctest3 /usr/bin/ctest

yum install -y redhat-lsb-core expat-devel libcurl-devel protobuf-compiler protobuf-devel protobuf rpmdevtools tar unzip ccache curl gcc gcc-c++ zlib-devel make git python2-devel python3-devel python3-numpy libunwind icu aria2 rsync python3-setuptools python3-wheel bzip2

