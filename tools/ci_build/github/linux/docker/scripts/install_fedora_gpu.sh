#!/bin/bash
set -e
dnf install -y --best --allowerasing redhat-lsb-core expat-devel libcurl-devel protobuf-compiler protobuf-devel protobuf compat-openssl10 lttng-ust libcurl openssl-libs krb5-libs libicu tar unzip ccache curl gcc gcc-c++ zlib-devel make git python2-devel python3-devel python3-numpy libunwind icu aria2 rsync python3-setuptools python3-wheel bzip2
#install cuda 10.1 and cudnn 7
aria2c -q -d /tmp 'http://developer.download.nvidia.com/compute/cuda/repos/fedora29/x86_64/cuda-repo-fedora29-10.1.243-1.x86_64.rpm'
dnf install -y /tmp/cuda-repo-fedora29-10.1.243-1.x86_64.rpm
dnf install -y cuda-10.1.243-1.x86_64
tar -zxvf cudnn-10.1-linux-x64-v7.6.4.38.tgz --strip 1 -C /usr/local
ldconfig -v /usr/local/cuda/lib64 /usr/local/lib64

