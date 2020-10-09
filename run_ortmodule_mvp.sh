#!/bin/bash

cur_dir=$(basename `pwd`)

if [[ ${cur_dir} != "RelWithDebInfo" ]]
then
    echo "Going to build folder"
    cd build/Linux/RelWithDebInfo
fi

echo "Exporting PYTHONPATH to use build dir as onnxruntime package"
export PYTHONPATH=/home/thiagofc/dev/github/onnxruntime/build/Linux/RelWithDebInfo/

echo "Copying PyTorch frontend source-code to build folder"
cp -Rf ../../../orttraining/orttraining/python/training/* ../../../build/Linux/RelWithDebInfo/onnxruntime/training/

echo "Running MNIST through Optimized API (ORTTrainer) to get full training graph"
python ../../../samples/python/mnist/ort_mnist.py --train-steps 1 --test-batch-size 0 --save-path "."

echo "Splitting full training graph in forward and backward graphs"
python ../../../orttraining/orttraining/test/python/orttraining_test_ortmodule_basic_transform_model.py model_with_training.onnx

echo "Running Flexible API (ORTModule)"
python ../../../orttraining/orttraining/test/python/orttraining_test_ortmodule_basic.py