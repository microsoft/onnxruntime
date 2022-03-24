# TVM Execution Provider

## Contents

- [Introduction](#introduction)
- [Build](#build-onnx-runtime-with-the-tvm-execution-provider)
- [Configuration options](#configuration-options)
- [Performance Tuning](#performance-tuning)
- [Samples](#samples)
- [Known issues](#known-issues)


## Introduction

TVM is an execution provider for ONNX Runtime that is built on top of Apache TVM. It enables ONNX Runtime users to leverage Apache TVM model optimizations.
TVM EP is currently in "Preview". It's been tested to work on a handful of models on Linux, but not on Windows or MacOS.

## Build ONNX Runtime with the TVM Execution Provider

Install the minimal pre-requisites on Ubuntu/Debian like linux operating systems:
```bash
apt-get install -y python3 python3-dev python3-pip python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev llvm-12
pip3 install numpy decorator attrs
```

Also, the current implementation has `NVidia GPU` support for TVM EP. For now, you can use only `NVidia GPU` with CUDA Toolkit support.
To do this, make sure you have installed the NVidia driver and CUDA Toolkit.
More detailed instructions can be found on the [official page](https://developer.nvidia.com/cuda-toolkit).

Clone this repo.
In order to build ONNXRT you will need to have CMake 3.18 or higher. In Ubuntu 20.04 you can use the following commands to install the latest version of CMake:

```bash
sudo apt-get update
sudo apt-get install gpg wget

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update

sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
sudo apt-get install kitware-archive-keyring

sudo apt-get install cmake
```

Build ONNX Runtime (TVM x86):
```bash
./build.sh --config Release --enable_pybind --build_wheel --parallel --skip_tests --skip_onnx_tests --use_tvm
```

Build ONNX Runtime (TVM with CUDA support):
```bash
./build.sh --config Release --enable_pybind --build_wheel --parallel --skip_tests --skip_onnx_tests --use_tvm --tvm_cuda_runtime
```

This command builds both `TVM` and `onnxruntime-tvm`. It creates two wheel, one for each project.
Build the python API for ONNX Runtime instead of using the standard package. Instructions for this are given below.

Package for TVM:
```bash
cd <path_to_onnx_runtime>
python3 -m pip uninstall tvm -y
whl_path=$(find ./build/<OS_NAME>/Release/_deps/tvm-src/python/dist -name "*.whl")
python3 -m pip install $whl_path
```

Package for TVM EP:
```bash
cd <path_to_onnx_runtime>
python3 -m pip uninstall onnxruntime onnxruntime-tvm -y
whl_path=$(find ./build/<OS_NAME>/Release/dist -name "*.whl")
python3 -m pip install $whl_path
```

Alternatively, you can set `PYTHONPATH` to tell python where to find the ONNXRT library and the TVM library.
```bash
export PYTHONPATH=<path_to_onnx_runtime>/build/<OS_NAME>/Release:${PYTHONPATH}
export PYTHONPATH=<path_to_onnx_runtime>/build/<OS_NAME>/Release/_deps/tvm-src/python:${PYTHONPATH}
```

## Configuration options
TVM Executor Provider can be configured with the following provider options:
```python
po = [dict(target=client_target,
           target_host=client_target_host,
           opt_level=client_opt_level,
           freeze_weights=freeze,
           tuning_file_path=client_tuning_logfile,
           input_names = input_names_str,
           input_shapes = input_shapes_str)]
tvm_session = onnxruntime.InferenceSession(model_path, providers=["TvmExecutionProvider"], provider_options=po)
```
<br>

- `target` and `target_host` are strings like in TVM (e.g. "llvm --mcpu=avx2"). When using accelerators, target may be something like `cuda` while target_host may be `llvm -mtriple=x86_64-linux-gnu`
- `opt_level` is TVM optimization level. It is 3 by default
- `freeze_weights` means that all model weights are kept on compilation stage otherwise they are downloaded each inference. True is recommended value for the best performance. It is true by default.
- `tuning_type` defines the type of TVM tuning logs being used, and can be set to either `AutoTVM` (1st gen auto tuning logs) or `Ansor` (2nd gen auto tuning logs). By default this option is set to `AutoTVM`.
- `tuning_file_path` is path to AutoTVM or Ansor tuning file which gives specifications for given model and target for the best performance. (See below for more details).

TVM supports models with fixed graph only. If your model has unknown dimensions in input shapes (excluding batch size) you must provide the shape using the `input_names` and `input_shapes` provider options. Below is an example of what must be passed to `provider_options`:
```python
input_names = "input_1 input_2"
input_shapes = "[1 3 224 224] [1 2]"
```

## Performance Tuning
TVM optimizes machine learning models through an automated tuning process that produces model variants specific to targeted hardware architectures.  This process also generates 'tuning logs' that the TVM EP relies on to maximize model performance. These logs can be acquired for your model by either using TVM as described here:

AutoTVM:
https://tvm.apache.org/docs/how_to/tune_with_autotvm/index.html

Ansor (Autoscheduling):
https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/index.html

or by using logs generated through the OctoML platform (https://onnx.octoml.ai) using instructions [here](https://help.octoml.ai/en/articles/5814452-using-octoml-platform-logs-with-onnx-rt-tvm-ep)

Using the TVM EP with TVM tuning logs also requires users to turn off ONNX Runtime preprocessing.  To do this, the following `SessionOptions()` can be used:
```
so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

tvm_session = onnxruntime.InferenceSession(model_path, sess_options=so, providers=["TvmExecutionProvider"], provider_options=po)
```

## Samples
- [Sample notebook for ResNet50 inference with TVM EP](https://github.com/microsoft/onnxruntime/blob/master/docs/python/inference/notebooks/onnxruntime-tvm-tutorial.ipynb)

## Known issues
- At this moment, the TVM EP has only been verified on UNIX/Linux systems.
- CUDA/GPU support is still in pre-alpha mode and results are expected to change. It is recommended that only CPU targets are used.
- Some compatibility issues have been found between ONNX and Google protobuf. `AttributeError: module 'google.protobuf.internal.containers' has no attribute 'MutableMapping'`. This usually occurss during `import onnx` in any python scripts for protobuf version >= 3.19.0 and ONNX version <= 1.8.1. To resolve the issue Google protobuf and ONNX can be reinstalled separately or together using:
```
pip3 uninstall onnx -y
pip3 install onnx==1.10.1
pip3 uninstall protobuf -y
pip3 install protobuf==3.19.1
```

The following pair of ONNX and protobuf versions have been found to be compatible:
- 3.17.3 and 1.8.0
- 3.19.1 and 1.10.1
