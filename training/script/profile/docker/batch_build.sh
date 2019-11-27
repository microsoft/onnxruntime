#!/bin/bash

current_dir=`pwd`
image_name="phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1"

echo "============"
docker run -it -v $current_dir:/scripts $image_name  bash /scripts/build.sh training 4e59cb6a
echo "==========="
docker run -it -v $current_dir:/scripts $image_name  bash /scripts/build.sh training ff12625d
echo "========="
docker run -it -v $current_dir:/scripts $image_name  bash /scripts/build.sh training 1a5fe050
echo "=========="
docker run -it -v $current_dir:/scripts $image_name  bash /scripts/build.sh wezhan/new-elementwise1 3c1e84b4