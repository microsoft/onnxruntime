#!/bin/bash
set -ex

pwd
echo $ORT_ROOT

files=($(find . -name "*.gcno"))
for my_path in "${files[@]}"
do
  parent_path=$( echo $my_path | grep -Eo "(onnxruntime/).*(.gcno)")
  if [ -n $parent_path ]; then
	  prefix_path=$( echo $my_path | grep -Eo ".*(.dir)" )
	  dest_dir="$ORT_ROOT"/"$prefix_path"/"$ORT_ROOT"
	  mkdir -p $dest_dir
	  # Mac doesn't support -Po
	  parent_dir=$(echo $my_path | grep -Eo ".*(onnxruntime/)")
	  parent_dir1=${parent_dir%%'onnxruntime/'}
	  pushd $parent_dir1
	  # https://stackoverflow.com/questions/11246070/cp-parents-option-on-mac
	  rsync -R "$parent_path" "$dest_dir"
	  popd
  else
      echo $my_path
  fi
done