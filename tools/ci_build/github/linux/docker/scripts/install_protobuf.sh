#!/bin/bash
set -e
GetFile https://github.com/protocolbuffers/protobuf/archive/v3.6.1.tar.gz /tmp/src/v3.6.1.tar.gz
tar -xf /tmp/src/v3.6.1.tar.gz -C /tmp/src
cd /tmp/src/protobuf-3.6.1
cmake ./cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Relwithdebinfo
make -j`nproc`
make install
