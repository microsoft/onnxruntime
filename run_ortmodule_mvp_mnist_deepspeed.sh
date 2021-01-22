#!/bin/bash

cur_dir=$(basename `pwd`)

if [[ ${cur_dir} != "RelWithDebInfo" ]]
then
    echo "Going to build folder (aka build/Linux/RelWithDebInfo)"
    cd build/Linux/RelWithDebInfo
fi

echo "Exporting PYTHONPATH to use build dir as onnxruntime package"
export PYTHONPATH=$(pwd)

echo "Copying PyTorch frontend source-code to build folder"
cp -Rf ../../../orttraining/orttraining/python/training/* ../../../build/Linux/RelWithDebInfo/onnxruntime/training/

echo "Running Flexible API (ORTModule)"
python ../../../orttraining/orttraining/test/python/orttraining_test_ortmodule_deepspeed_zero_stage_1.py --help
deepspeed ../../../orttraining/orttraining/test/python/orttraining_test_ortmodule_deepspeed_zero_stage_1.py --deepspeed_config ../../../orttraining/orttraining/test/python/orttraining_test_ortmodule_deepspeed_zero_stage_1_config.json $@
