#!/bin/bash
# according to https://github.com/actions/virtual-environments/issues/5879
set -ex
if [[ "$(uname -s)" == 'Darwin' ]]; then
    function get_full_ndk_version {
        majorVersion=$1
        ndkVersion=$(${SDKMANAGER} --list | grep "ndk;${majorVersion}.*" | awk '{gsub("ndk;", ""); print $1}' | sort -V | tail -n1)
        echo "$ndkVersion"
    }
    ANDROID_HOME=$HOME/Library/Android/sdk
    SDKMANAGER=$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager
    ndkDefault=$(get_full_ndk_version "23")
    ln -s $ANDROID_HOME/ndk/$ndkDefault $ANDROID_HOME/ndk-bundle
fi

if [[ "$(uname -s)" == 'Linux' ]]; then
    function get_full_ndk_version {
        majorVersion=$1
        ndkFullVersion=$($SDKMANAGER --list | grep "ndk;${majorVersion}.*" | awk '{gsub("ndk;", ""); print $1}' | sort -V | tail -n1)
        echo "$ndkFullVersion"
    }
    ANDROID_ROOT=/usr/local/lib/android
    ANDROID_SDK_ROOT=${ANDROID_ROOT}/sdk
    ANDROID_NDK_ROOT=${ANDROID_SDK_ROOT}/ndk-bundle
    SDKMANAGER=${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin/sdkmanager
    ndkDefaultFullVersion=$(get_full_ndk_version "23")
    ln -sf $ANDROID_SDK_ROOT/ndk/$ndkDefaultFullVersion $ANDROID_NDK_ROOT
fi
