#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    DIR_OS="MacOS"
else
    DIR_OS="Linux"
fi

if [[ "$*" == *"--ios"* ]]; then
    DIR_OS="iOS"
elif [[ "$*" == *"--android"* ]]; then
    DIR_OS="Android"
fi

# Telemetry uses the 1DS SDK, which is not supported for WebAssembly/Emscripten builds.
# Only request it for native builds so that `./build.sh --build_wasm` keeps working without
# the user having to override the wrapper's default.
TELEMETRY_ARG="--use_telemetry"
if [[ "$*" == *"--build_wasm"* ]]; then
    TELEMETRY_ARG=""
fi

python3 $DIR/tools/ci_build/build.py --build_dir $DIR/build/$DIR_OS $TELEMETRY_ARG "$@"
