#!/bin/bash

# This script will setup gradlew to use gradle version 6.8.3 for Android CI,
# since the macOS pipeline is using gradle 7.0 which will fail the java build
# See, https://github.com/actions/virtual-environments/issues/3195

set -e
set -x

if [ $# -ne 1 ]; then
    echo "One command line argument, the ORT root directory, is expected"
fi

ORT_ROOT=$1

pushd ${ORT_ROOT}/java
gradle wrapper --gradle-version 6.8.3 --no-daemon --no-watch-fs
./gradlew --version
popd
