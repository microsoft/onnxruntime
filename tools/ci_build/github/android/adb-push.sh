#! /usr/bin/env bash
# Created by daquexian
# A bash script that push folders recursively, "adb push" doesn't work on some devices

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 src dest"
    exit 1
fi

src=`realpath $1`
src_basename=`basename $1`
pushd `dirname $src`
if [[ -d $src ]]; then
    find $src_basename -type d -print -exec adb shell mkdir -p $2/{} \; > /dev/null
fi
find $src_basename -type f -exec adb push {} $2/{} \; > /dev/null
popd
