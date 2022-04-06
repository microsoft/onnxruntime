# onnxflow


1. Build onnxruntime with on device training flag:
```sh
./build.sh --config RelWithDebInfo --enable_training --use_cuda --cuda_home /usr/local/cuda/ --cudnn_home /usr/local/cuda/ --build_wheel --parallel --cuda_version=11.3 --skip_tests --build_wheel --build_on_device_training
```

This will generate the protobuf files needed for serialization and deserialization of the parameters:
- orttraining/orttraining/onnxflow/csrc/onnxflow.pb.h
- orttraining/orttraining/onnxflow/csrc/onnxflow.pb.cc
- orttraining/orttraining/onnxflow/onnxflow/onnxflow_pb2.py

2. Compose the model with the necessary loss and optimizer by running this from `orttraining/orttraining/onnxflow`:
```py
python sample.py
```

This will create the following:
- Forward+Loss+Backward training onnx graph
- Optimizer onnx graph
- Serialized parameters (saved as `parameters.of`)

3. Use the saved onnx files and the parameters to perform training. To load the serialized parameters, see example utility `orttraining/orttraining/onnxflow/sample.m.cpp`. And run it by executing
```sh
orttraining_on_device_sample
```
Pass in the absolute path of the `parameters.of` file when prompted.
