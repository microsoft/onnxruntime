#!/bin/bash
set -e
dnf install -y binutils gcc gcc-c++ aria2 python3-pip python3-wheel git python3-devel cmake
python3 -m pip install --upgrade pip
python3 -m pip install numpy
