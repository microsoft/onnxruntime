#!/bin/bash
set -e
dnf install -y --best --allowerasing rpmdevtools tar unzip ccache curl gcc gcc-c++ zlib-devel make git python2-devel python3-devel python3-numpy libunwind icu aria2 rsync python3-setuptools python3-wheel bzip2
