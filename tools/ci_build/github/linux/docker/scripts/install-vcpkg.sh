#!/bin/bash
set -e -x
mkdir -p $VCPKG_INSTALLATION_ROOT 
cd $VCPKG_INSTALLATION_ROOT 
curl -O -sSL https://github.com/microsoft/vcpkg/archive/refs/heads/master.tar.gz 
tar --strip=1 -zxf master.tar.gz
./bootstrap-vcpkg.sh 
chmod -R 0777 $VCPKG_INSTALLATION_ROOT
