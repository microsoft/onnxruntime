#!/bin/bash 

build_dir=${1:-"."}
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

gtest_filter="-"
while read line; do
  gtest_filter="$gtest_filter:$line"
done <$script_dir/pai-excluded-tests.txt

$build_dir/onnxruntime_test_all --gtest_filter=$gtest_filter
