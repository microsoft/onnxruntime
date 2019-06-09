#!/bin/bash
set -e
dnf install -y --best --allowerasing libjpeg-turbo-devel redhat-lsb-core expat-devel libcurl-devel protobuf-compiler protobuf-devel protobuf rpmdevtools tar unzip ccache curl gcc gcc-c++ zlib-devel make git python2-devel python3-devel python3-numpy libunwind icu aria2 rsync python3-setuptools python3-wheel bzip2
