# Standalone TVM (STVM) Execution Provider

## Contents

- [Introduction](#introduction)
- [Build](#build)
- [Configuration options](#configuration-option)
- [Performance Tuning](#performance-tuning)
- [Samples](#samples)
- [Known issues](#known-issues)


## Introduction

STVM is an execution provider for ONNX Runtime that is built on top of Apache TVM. It enables ONNX Runtime users to leverage Apache TVM model optimizations.
STVM EP is currently in "Preview". It's been tested to work on a handful of models on Linux, but not on Windows or MacOS.

### Build ONNX Runtime with the STVM Execution Provider

Install the minimal pre-requisites on Ubuntu/Debian like linux operating systems:
```
apt-get install -y python3 python3-dev python3-pip python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev llvm-12
pip3 install numpy decorator attrs
```

Clone this repo.

In order to build ONNXRT you will need to have CMake 3.18 or higher. In Ubuntu 20.04 you can use the following commands to install the latest version of CMake:

```
sudo apt-get update
sudo apt-get install gpg wget

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update

sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
sudo apt-get install kitware-archive-keyring

sudo apt-get install cmake
```

Build ONNX Runtime:
```
./build.sh --config Release --enable_pybind --build_wheel --skip_tests --parallel --use_stvm --skip_onnx_tests
```

This command builds both TVM and onnxruntime-stvm. It creates two wheel, one for each project.
Build the python API for ONNX Runtime instead of using the standard package:
```
cd <path_to_onnx_runtime>
pip3 uninstall onnxruntime onnxruntime-stvm tvm -y
whl_path=$(find ./build/Linux/Release/dist -name "*.whl")
python3 -m pip install $whl_path
```
Alternatively, you can set PYTHONPATH to tell python where to find the ONNXRT library and the TVM library.
```
export PYTHONPATH=$ORT_PYTHON_HOME:$TVM_PYTHON_HOME:${PYTHONPATH}
```

## Configuration options
STVM Executor Provider can be configured with the following provider options:
```python
po = [dict(target=client_target,
           target_host=client_target_host,
           opt_level=client_opt_level,
           freeze_weights=freeze,
           tuning_file_path=client_tuning_logfile,
           input_names = input_names_str,
           input_shapes = input_shapes_str)]
stvm_session = onnxruntime.InferenceSession(model_path, providers=["StvmExecutionProvider"], provider_options=po)
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
TVM optimizes machine learning models through an automated tuning process that produces model variants specific to targeted hardware architectures.  This process also generates 'tuning logs' that the STVM EP relies on to maximize model performance. These logs can be acquired for your model by either using TVM as described here:

AutoTVM:
https://tvm.apache.org/docs/how_to/tune_with_autotvm/index.html

Ansor (Autoscheduling):
https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/index.html

or by using logs generated through the OctoML platform (https://onnx.octoml.ai) using instructions [here](https://help.octoml.ai/en/articles/5814452-using-octoml-platform-logs-with-onnx-rt-tvm-ep)

Using the STVM EP with TVM tuning logs also requires users to turn off ONNX Runtime preprocessing.  To do this, the following `SessionOptions()` can be used:
```
so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

stvm_session = onnxruntime.InferenceSession(model_path, sess_options=so, providers=["StvmExecutionProvider"], provider_options=po)
```

## Samples
- [Sample notebook for ResNet50 inference with STVM EP](https://github.com/octoml/onnxruntime/blob/STVM_EP_PR/docs/python/inference/notebooks/onnxruntime-stvm-tutorial.ipynb)

## Known issues
- At this moment, the STVM EP has only been verified on UNIX/Linux systems.
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

When use onnxruntime-stvm after it was build from the source, the following error may happen:

```
terminate called after throwing an instance of 'tvm::runtime::InternalError'
  what():  [12:01:11] ..._deps/tvm-src/src/runtime/registry.cc:69: 
---------------------------------------------------------------
An error occurred during the execution of TVM.
For more information, please see: https://tvm.apache.org/docs/errors.html
---------------------------------------------------------------
  Check failed: (can_override) is false: Global PackedFunc arith.CreateAnalyzer is already registered

Aborted (core dumped)
```

It means both onnxruntime and tvm loaded a different dynamic library ``libtvm.[so|dll]``.
To solve that, `tvm` must be imported first:

```python
import tvm
import onnxruntime
```
