#!/bin/bash

build_dir=${1:-"."}
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Warning: The following tests are EXCLUDED on PAI agent:"
gtest_filter="-"
while read line; do
  gtest_filter="$gtest_filter:$line"
  echo "$line"
done <$script_dir/pai-excluded-tests.txt
echo ""

echo "Running ./onnxruntime_test_all .."
$build_dir/onnxruntime_test_all --gtest_filter=$gtest_filter

echo "Running ./onnxruntime_provider_test .."
$build_dir/onnxruntime_provider_test --gtest_filter=$gtest_filter
