#!/bin/bash
set -e

yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

yum install -y redhat-lsb-core expat-devel libcurl-devel rpmdevtools tar unzip ccache curl gcc gcc-c++ zlib-devel make  python2-devel python3-devel python3-pip python3-numpy libunwind icu aria2 rsync python3-setuptools python3-wheel bzip2 git

