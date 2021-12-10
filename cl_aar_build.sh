#!/bin/bash

set -ev

export ANDROID_HOME=`realpath ~/android_sdk/`
export ANDROID_NDK_HOME=`realpath ~/android_sdk/ndk/23.0.7599858`

python tools/ci_build/github/android/build_aar_package.py     \
    --config Release                                          \
    --build_dir build_aar                                     \
    tools/ci_build/github/android/cl_aar_build_settings.json
