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

cd "$BINARY_DIR/onnxruntime-java"
rm -f *.asc
rm -f *.sha256
rm -f *.sha512
rm -f *.pom
ls
cd ..
mkdir tests
cd tests
jar xf ../onnxruntime-java/testing.jar
rm -f ../onnxruntime-java/testing.jar
echo "Java Version"
java -version

echo "Directories created"
echo  "Library path:" "$LD_LIBRARY_PATH"

java -DUSE_CUDA=1 -cp "$BINARY_DIR/tests:$BINARY_DIR/onnxruntime-java/*" org.junit.platform.console.ConsoleLauncher --scan-classpath=$BINARY_DIR/tests \
            --fail-if-no-tests --disable-banner
