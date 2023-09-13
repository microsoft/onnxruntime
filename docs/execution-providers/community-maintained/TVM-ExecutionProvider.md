---
title: Apache - TVM
description: Instructions to execute ONNX Runtime with the Apache TVM execution provider
grand_parent: Execution Providers
parent: Community-maintained
nav_order: 3
---

# TVM Execution Provider
{: .no_toc }

TVM is an execution provider for ONNX Runtime that is built on top of Apache TVM. It enables ONNX Runtime users to leverage Apache TVM model optimizations.
TVM EP is currently in "Preview". It's been tested to work on a handful of models on Linux and Windows, but not on MacOS.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Build ONNX Runtime with the TVM Execution Provider

### **Linux**
Install the minimal pre-requisites on Ubuntu/Debian like linux operating systems:
```bash
apt-get install -y python3 python3-dev python3-pip python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev llvm-12
pip3 install numpy decorator attrs nasm
```
**Note**: since ONNX Runtime with TVM EP is built with Intel ipp-crypto library there are new requirements. Compiler gcc (and g++) version should be equal to or higher than 8.2. nasm version should be 2.14.02 or higher. Problem with small nasm version can be seen [here](https://github.com/intel/ipp-crypto/issues/9) or [here](https://bugzilla.nasm.us/show_bug.cgi?id=3392205). For ubuntu LTS 18 `apt-get install nasm` is not enough due to it has version 2.13.02, see how to install from sources instruction [here](https://stackoverflow.com/questions/36144930/steps-to-install-nasm-offline-on-ubuntu).

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

### **Windows**
Install the minimal prerequisites on Windows: Git, CMake, Visual Studio, Python, LLVM
- Git: Download Git for Windows from [here](https://git-scm.com/download/win) and install it. Please make sure that the git.exe path is included in the environment variable. By default, it should be added. To check git after the installation use `git --version` in command line (cmd).
- CMake: use [the link](https://cmake.org/download/) to download and install CMake. msi-file is recommended for it. To verify CMake installation use `cmake --version` in cmd.
- Visual Studio: Download from [here](https://visualstudio.microsoft.com/ru/downloads/) and install Visual Studio 20** Community & Visual Studio Build Tools respectively. It is recommended not to change the default installation path. Chose "Desktop development with C++" workload and make sure that both options of “MSVC [contemporary version] C++ build tools” and “Windows 10 SDK” are selected.
- Python: Download Python 3.* from [here](https://www.python.org/downloads/windows/) and install it. Please have a check on the option of “Add Python to PATH”, so the installer will include the Python directory into the environment variable directly. To check python after the installation use `python` from cmd. The expected output is similar to the following:
```cmd
Python 3.10.5 (tags/v3.10.5:f377153, Jun  6 2022, 16:14:13) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```
Use `quit()` to exit from python interface.
- LLVM: the compiler is not necessary for pure ONNX Runtime installation but it is needed for TVM EP by default.
```cmd
git clone --depth 1 --branch release/11.x https://github.com/llvm/llvm-project.git
cmake -S llvm -B build -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi" -DLLVM_TARGETS_TO_BUILD=X86 -Thost=x64 -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022"
cmake --build ./build --config Release
```
- Dependencies of ipp-crypto:<br>
1. install asm compiler (nasm) on windows by line:
```cmd
winget install nasm -i
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Add it to PATH (instruction for Windows GUI can be seen [here](https://www.computerhope.com/issues/ch000549.htm#dospath)) or by cmd:
```
set PATH="%PATH%;C:\Program Files\NASM"
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Check by `nasm --version` in prompt command line.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
2. install openssl on windows by msi-file from [here](https://slproweb.com/products/Win32OpenSSL.html)
Add path to directory (e.g. "C:\Program Files\OpenSSL-Win64\bin") with executable file to PATH (see instructions above).<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Check by `openssl version` in prompt command line.
<br>
<br>

For using NVIDIA GPU (optional) CUDA and cuDNN should be installed.
- CUDA: Install CUDA by the [link](https://developer.nvidia.com/cuda-11.0-download-archive).
- cuDNN: download cuDNN installer from [here](https://developer.nvidia.com/rdp/cudnn-archive). Choose v8.* for corresponding CUDA v11.*, unzip it, and move cuDNN files as following:
1. [unzipped dir]\bin\ → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin
2. [unzipped dir]\include\ → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include
3. [unzipped dir]\lib\ → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib

To verify the CUDA installation use `nvcc --version` in cmd.
<br>
<br>

Build ONNX Runtime with TVM Execution Provider from source:
- Use command line and clone sources from github:
```cmd
git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime
```
- CPU build:
```
build.bat --config Release --enable_pybind --build_wheel --skip_tests --parallel --use_tvm --skip_onnx_tests --cmake_generator "Visual Studio 17 2022" --llvm_config <path_to_llvm_root>/build/Release/bin/llvm-config.exe
```
- GPU build:
```
build.bat --config Release --enable_pybind --build_wheel --skip_tests --parallel --use_tvm --skip_onnx_tests --cmake_generator "Visual Studio 17 2022" --llvm_config <path_to_llvm_root>/build/Release/bin/llvm-config.exe --use_cuda --cudnn_home “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.*” --cuda_home “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.*”
```
In both cases (CPU, GPU) there are the following options for cmake generator: "Visual Studio 15 2017", "Visual Studio 16 2019", "Visual Studio 17 2022" and "Ninja"
- Install python wheel package for ONNX Runtime:<br>
Default path to the package is `<path_to_onnxruntime_root>/build/Windows/Release/Release/dist`. Note that it is different in comparison with path to the package on Linux. Before installation check names of wheel packages and use corresponding one. It can be looked like the following:
```cmd
python -m pip install .\onnxruntime\build\Windows\Release\Release\dist\onnxruntime_tvm-1.6.0-cp37-cp37m-win_amd64.whl
```
- Install python wheel package for TVM due to its python API is used inside TVM EP:<br>
It can be looked like the following:
```cmd
python -m pip install .\onnxruntime\build\Windows\Release\_deps\tvm-src\python\dist\tvm-0.9.dev1728+g3425ed846-cp39-cp39-win_amd64.whl
```
- Verify result by python script. Note: python should not be launched from directory containing 'onnxruntime' directory for correct result:
```python
import onnxruntime
print(onnxruntime.__version__)
print(onnxruntime.get_device())
print(onnxruntime.get_available_providers())
```
- Uninstall procedure:
```cmd
pip uninstall onnxruntime-tvm
```

## Configuration options
TVM Executor Provider can be configured with the following provider options:
```python
po = [dict(executor=tvm_executor_type,
           so_folder=folder_with_pretuned_files,
           check_hash=check_hash,
           hash_file_path=hash_file_path,
           target=client_target,
           target_host=client_target_host,
           opt_level=client_opt_level,
           freeze_weights=freeze,
           to_nhwc=layout_transform,
           tuning_type=tvm_optimizer_type,
           tuning_file_path=client_tuning_logfile,
           input_names = input_names_str,
           input_shapes = input_shapes_str)]
tvm_session = onnxruntime.InferenceSession(model_path, providers=["TvmExecutionProvider"], provider_options=po)
```
<br>

- `executor` is executor type used by TVM. There is choice between two types: GraphExecutor and VirtualMachine which are corresponded to "graph" and "vm" tags. VirtualMachine is used by default.
- `so_folder` is path to folder with set of files (.ro-, .so/.dll-files and weights) obtained after model tuning. It uses these files for executor compilation instead of onnx-model. But the latter is still needed for ONNX Runtime.
- `check_hash` means that it is necessary to perform a HASH check for the model obtained in the `so_folder` parameter. It is `False` by default.
- `hash_file_path` is path to file that contains the pre-computed HASH for the ONNX model which result of tuning locates in the path passed by `so_folder` parameter.
  If an empty string was passed as this value, then the file will be searched in the folder that was passed in the `so_folder` parameter.
- `target` and `target_host` are strings like in TVM (e.g. "llvm --mcpu=avx2"). When using accelerators, target may be something like `cuda` while target_host may be `llvm -mtriple=x86_64-linux-gnu`
- `opt_level` is TVM optimization level. It is 3 by default
- `freeze_weights` means that all model weights are kept on compilation stage otherwise they are downloaded each inference. True is recommended value for the best performance. It is true by default.
- `to_nhwc` switches on special model transformations, particularly data layout, which Octomizer is used. It allows to work correctly with tuning logs obtained from Octomizer. It is false by default.
- `tuning_type` defines the type of TVM tuning logs being used, and can be set to either `AutoTVM` (1st gen auto tuning logs) or `Ansor` (2nd gen auto tuning logs). By default this option is set to `AutoTVM`.
- `tuning_file_path` is path to AutoTVM or Ansor tuning file which gives specifications for given model and target for the best performance. (See below for more details).

TVM supports models with fixed graph only. If your model has unknown dimensions in input shapes (excluding batch size) you must provide the shape using the `input_names` and `input_shapes` provider options. Below is an example of what must be passed to `provider_options`:
```python
input_names = "input_1 input_2"
input_shapes = "[1 3 224 224] [1 2]"
```

## Performance Tuning
TVM optimizes machine learning models through an automated tuning process that produces model variants specific to targeted hardware architectures.  This process also generates 'tuning logs' that the TVM EP relies on to maximize model performance. These logs can be acquired for your model either by using TVM as described here:

AutoTVM:
https://tvm.apache.org/docs/how_to/tune_with_autotvm/index.html

Or

Ansor (Autoscheduling):
https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/index.html

Using the TVM EP with TVM tuning logs also requires users to turn off ONNX Runtime preprocessing.  To do this, the following `SessionOptions()` can be used:
```python
so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

tvm_session = onnxruntime.InferenceSession(model_path, sess_options=so, providers=["TvmExecutionProvider"], provider_options=po)
```

### **Using precompiled model**
It is also possible to use a precompiled model.

The compiled model can be obtained using the [OctoML platform](https://onnx.octoml.ai)
or compiled directly (see **Support precompiled model** section in
[Sample notebook for ResNet50 inference with TVM EP](https://github.com/microsoft/onnxruntime/blob/main/docs/python/notebooks/onnxruntime-tvm-tutorial.ipynb)
for more information on model compilation).

In order to use the precompiled model, only need to pass two options:
* **executor** - `vm` (`VirtualMachine`) must be used as a value
(this functionality is not supported for `GraphExecutor`);
* **so_folder** - as a value, you must pass the path to the directory where
the files of the precompiled model are located.
* **check_hash** - (optional) if you want to check hash, you must pass `True` as the value.
* **hash_file_path** - (optional) by default, the file containing the hash for the tuned model will be searched in the directory that is passed in the `so_folder` parameter. If you want to specify different location, then you must pass the path to the file that contains the desired hash as a value.

You can read more about these options in section [Configuration options](#configuration-options) above.

## Samples
- [Sample notebook for ResNet50 inference with TVM EP](https://github.com/microsoft/onnxruntime/blob/main/docs/python/notebooks/onnxruntime-tvm-tutorial.ipynb)

## Known issues
- At this moment, the TVM EP has only been verified on UNIX/Linux and Windows systems.
- Some compatibility issues have been found between ONNX and Google protobuf. `AttributeError: module 'google.protobuf.internal.containers' has no attribute 'MutableMapping'`. This usually occurss during `import onnx` in any python scripts for protobuf version >= 3.19.0 and ONNX version <= 1.8.1. To resolve the issue Google protobuf and ONNX can be reinstalled separately or together using:
```bash
pip3 uninstall onnx -y
pip3 install onnx==1.10.1
pip3 uninstall protobuf -y
pip3 install protobuf==3.19.1
```

The following pair of ONNX and protobuf versions have been found to be compatible:
- 3.17.3 and 1.8.0
- 3.19.1 and 1.10.1
