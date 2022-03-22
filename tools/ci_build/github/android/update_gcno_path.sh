#!/bin/bash
set -x

pwd
echo $ORT_ROOT
echo $FILE_TYPE

files=($(find . -name "*.$FILE_TYPE"))
for my_path in "${files[@]}"
do
  old_root=$(echo $my_path | grep -Eo "(/mnt/.*/s/)")
  if [ -z "$old_root" ]; then
      echo $my_path
      continue
  fi
  # Mac doesn't support -Po, we could use (?=)
  file_with_parent=$(echo $my_path | grep -Eo "(($old_root).*)")
  file_with_parent=${file_with_parent##$old_root}
  if [ -n "$file_with_parent" ]; then
	  old_dir=$(echo $my_path | grep -Eo "(.*($old_root))")
	  prefix_path=${old_dir%%$old_root}
	  dest_dir="$ORT_ROOT"/"$prefix_path"/"$ORT_ROOT"
	  mkdir -p $dest_dir

	  if [ -n "$old_dir" ]; then
		  pushd $old_dir
	      # https://stackoverflow.com/questions/11246070/cp-parents-option-on-mac
	      rsync --remove-source-files -R "$file_with_parent" "$dest_dir"
		  popd
	  else
	      echo $my_path
	  fi
  else
      echo $my_path
  fi
done