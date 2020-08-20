#!/bin/bash

# --------------------------------------------------------------
# Copyright(C) Xilinx Inc.
# Licensed under the MIT License.
# --------------------------------------------------------------

user=`whoami`
uid=`id -u`
gid=`id -g`

xclmgmt_driver="$(find /dev -name xclmgmt\*)"
docker_devices=""
for i in ${xclmgmt_driver} ;
do
  docker_devices+="--device=$i "
done

render_driver="$(find /dev/dri -name renderD\*)"
for i in ${render_driver} ;
do
  docker_devices+="--device=$i "
done

docker run \
    $docker_devices \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -v $PWD:/workspace \
    -w /workspace \
    -it \
    --rm \
    --network=host \
    onnxruntime-vitisai
