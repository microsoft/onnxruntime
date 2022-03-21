#!/bin/bash
set -ex

pwd

files=$(find . -name "*.gcno")
for my_path in "${files[@]}"
do 
  parent_path=$( echo $my_path | grep -Eo "(onnxruntime/).*(.gcno)")
  prefix_path=$( echo $my_path | grep -Eo ".*(.dir)" )
  mkdir -p "$prefix_path"/"$(Build.SourcesDirectory)"
  cp --parents parent_path "$prefix_path"/"$(Build.SourcesDirectory)"/
done