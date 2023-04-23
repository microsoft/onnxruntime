#!/bin/bash

# https://py-pkgs.org/04-package-structure.html#package-installation

if [ -z "$1" ]; then
  printf "Usage:\n\t $0 <dst>\n"
  exit -1
fi

version="dev"
dist_info_dir=$1/onnxruntime-${version}.dist-info

set -ex

mkdir -p ${dist_info_dir}

touch ${dist_info_dir}/INSTALLER
touch ${dist_info_dir}/LICENSE
touch ${dist_info_dir}/RECORD
touch ${dist_info_dir}/REQUESTED
touch ${dist_info_dir}/WHEEL

cat << EOF > ${dist_info_dir}/METADATA
Metadata-Version: 2.1
Name: onnxruntime
Version: dev
EOF
