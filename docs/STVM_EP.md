# Standalone TVM (STVM) Execution Provider

## Contents

- [Introduction](#introduction)
- [Build](#build)
- [Configuration options](#configuration-option)
- [Performance Tuning](#performance-tuning)
- [Samples](#samples)
- [Known issues](#known-issues)


## Introduction
STVM is an execution provider in the ONNX Runtime, which is built on top of Apache TVM. It delivers the path to innovations happening in Apache TVM ecosystem to ONNX Runtime customers.

## Build
There are two major steps to build ONNX runtime with STVM EP. Firstly, Apache TVM should be built, and then ONNX runtime.

Important note is that both TVM and ORT with STVM use Python API, therefore the python packages shoud be reinstall or PYTHONPATH should be updated accordingly for correct work. See details below:

### Prerequisites
Initially TVM and required dependencies should be installed:<br />

TVM dependencies installation:<br />
`apt-get install -y python3 python3-dev python3-pip python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev llvm`<br />
`pip3 install numpy decorator attrs`<br />
TVM installation from tvm_update folder:
```
cd onnxruntime/cmake/external/tvm_update/
mkdir build
cd ./build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_LLVM=ON (or -DUSE_CUDA=ON) ..
make -j8
```
Add correct python path to environment for TVM python API:
```
export TVM_HOME=<path_to_msft_onnxrt>/onnxruntime/cmake/external/tvm_update
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```
More detailed Apache TVM install instructions are located [here](https://tvm.apache.org/docs/install/from_source.html)

### Build ONNX runtime with STVM Execution Provider
Build onnxruntime:
```
./build.sh --config Release --enable_pybind --build_wheel --skip_tests --parallel --use_stvm --skip_onnx_tests
```
Build python API for onnxruntime instead of current one from standard package. This step can be skipped if shell variables export is used (see below)
```
pip3 uninstall onnxruntime onnxruntime-stvm -y
whl_path=$(find ./onnxruntime/build/Linux/Release/dist -name "*.whl")
python3 -m pip install $whl_path
```
It can be used instead of whl-package install:
```
export ORT_PYTHON_HOME=<path_to_msft_onnxrt>/onnxruntime/build/Linux/Release
export PYTHONPATH=$ORT_PYTHON_HOME:${PYTHONPATH}
```

## Configuration options
STVM Executor Provider can be configured with the following provider options:
```
po = [dict(target=client_target,
           target_host=client_target_host,
           opt_level=client_opt_level,
           freeze_weights=freeze,
           tuning_file_path=client_tuning_logfile,
           input_names = input_names_str,
           input_shapes = input_shapes_str)]
stvm_session = onnxruntime.InferenceSession(model_path, providers=["StvmExecutionProvider"], provider_options=po)
```
- `target` and `target_host` are strings like in TVM (e.g. "llvm --mcpu=avx2")
- `opt_level` is TVM optimization level. It is 3 by default
- `freeze_weights` means that all model weights are kept on compilation stage otherwise they are downloaded each inference. True is recommended value for the best performance. It is true by default.
- `tuning_file_path` is path to AutoTVM tuning file which gives specifications for given model and target for the best performance.
- TVM supports models with fixed graph only. If you have model with unknown dimensions in input shapes (excluding batch size) you can insert fixed values by `input_names` and `input_shapes` provider options. Due to specific of provider options parser inside ORT they are string with the following format:
```
input_names = "input_1 input_2"
input_shapes = "[1 3 224 224] [1 2]"
```

## Performance Tuning
As it was said above for the best model performance the tuning log file can be used. But due to some preprocessing of onnx model inside ONNX runtime before TVM gets it there can be differences between tuned model and obtained one. To turn off ONNX runtime preprocessing stage the session options can be used:
```
so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

stvm_session = onnxruntime.InferenceSession(model_path, sess_options=so, providers=["StvmExecutionProvider"], provider_options=po)
```

## Samples
- [Sample notebook for ResNet50 inference with STVM EP](https://github.com/octoml/onnxruntime/blob/STVM_EP_PR/docs/python/inference/notebooks/onnxruntime-stvm-tutorial.ipynb)

## Known issues
- At this moment, EP was verified only on UNIX/Linux systems
- There can be issue related to compatibility of ONNX and Google protobuf. `AttributeError: module 'google.protobuf.internal.containers' has no attribute 'MutableMapping'` error can be caught during `import onnx` in any python scripts for protobuf version >= 3.19.0 and onnx version <= 1.8.1. To resolve the issue Google protobuf and onnx can be reinstalled separately or together due to conflict between onnx and protobuf versions:
```
pip3 uninstall onnx -y
pip3 install onnx==1.10.1
pip3 uninstall protobuf -y
pip3 install protobuf==3.19.1
```
Two pairs of compartible protobuf and onnx versions: 3.17.3 and 1.8.0, and 1.19.1 and 1.10.1, correspondingly.