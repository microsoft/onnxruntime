#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This is for testing GPU final jar on Linux
set -e -o -x

usage() { echo "Usage: $0 [-r <binary directory>] [-v <version number>]" 1>&2; exit 1; }

while getopts r:v: parameter_Option
do case "${parameter_Option}"
in
r) BINARY_DIR=${OPTARG};;
v) VERSION_NUMBER=${OPTARG};;
*) usage ;;
esac
done

EXIT_CODE=1

uname -a

cd "$BINARY_DIR/final-jar"

mkdir test

echo "Directories created"
echo  "Library path:" "$LD_LIBRARY_PATH"

pushd test
jar xf "$BINARY_DIR/final-jar/testing.jar"
popd

curl -O -sSL https://oss.sonatype.org/service/local/repositories/releases/content/org/junit/platform/junit-platform-console-standalone/1.6.2/junit-platform-console-standalone-1.6.2.jar
curl -O -sSL https://oss.sonatype.org/service/local/repositories/releases/content/com/google/protobuf/protobuf-java/3.21.7/protobuf-java-3.21.7.jar
java -DUSE_CUDA=1 -jar ./junit-platform-console-standalone-1.6.2.jar -cp .:./test:./protobuf-java-3.21.7.jar:./onnxruntime_gpu-"${VERSION_NUMBER}".jar --scan-class-path --fail-if-no-tests --disable-banner


EXIT_CODE=$?

set -e
exit $EXIT_CODE
