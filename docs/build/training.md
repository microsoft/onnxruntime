---
title: Build for training
parent: Build ONNX Runtime
description: Learn how to build ONNX Runtime for training from source for different scenarios and hardware targets
nav_order: 2
redirect_from: /docs/how-to/build/training
---

# Build for On-Device Training
{: .no_toc }

## Prerequisites

- Python 3.x
- CMake

## Build Instructions for the Training Phase

1. Clone the repository

    ```bash
    git clone --recursive https://github.com/Microsoft/onnxruntime.git
    cd onnxruntime
    ```

2. Build ONNX Runtime for `On-Device Training`

   a. For Windows

    ```powershell
    .\build.bat --config RelWithDebInfo --cmake_generator "Visual Studio 17 2022" --build_shared_lib --parallel --enable_training_apis
    ```

   b. For Linux

    ```bash
    ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --enable_training_apis
    ```

   c. For Android

   Refer to the [Android build instructions](./android.md) and add the ```--enable_training_apis``` build flag.

   d. For MacOS

   Refer to the [macOS inference build instructions](./inferencing.md) and add the `--enable_training_apis` build flag.

   e. For iOS

    Refer to the [iOS build instructions](./ios.md) and add the ```--enable_training_apis``` build flag.

> **Note**
>
> - To build the C# bindings, add the ```--build_nuget``` flag to the build command above.
>
> - To build the Python wheel:
>   - add the ```--build_wheel``` flag to the build command above.
>   - install the wheel using ```python -m pip install build/Linux/RelWithDebInfo/dist/*.whl```
>
> - The ```config``` flag can be one of ```Debug```, ```RelWithDebInfo```, ```Release```, ```MinSizeRel```. Use the one that suits your use case.
>
> - The ```--enable_training_apis``` flag can be used in conjunction with the ```--minimal_build``` flag.
>
> - The offline phase of generating the training artifacts can only be done with Python (using the ```--build_wheel``` flag).
>
> - The build commands above only build for the cpu execution provider. To build for cuda execution provider, add these flags
>   - ```--use_cuda```
>   - ```--cuda_home {directory to your cuda home, for example /usr/local/cuda/}```
>   - ```--cudnn_home {directory to your cuda home, for example /usr/local/cuda/}```
>   - ```--cuda_version={version for example 11.8}```

# Build for Large Model Training
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

#### Linux

```
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --enable_training
```

## GPU / CUDA
### Prerequisites
{: .no_toc }

The default NVIDIA GPU build requires CUDA runtime libraries installed on the system:

* [CUDA](https://developer.nvidia.com/cuda-toolkit)
* [cuDNN](https://developer.nvidia.com/cudnn)

### Build instructions
{: .no_toc }

1. Checkout this code repo with

    ```bash
    git clone https://github.com/microsoft/onnxruntime
    cd onnxruntime
    ```

2. Set the environment variables: *adjust the paths for locations on your build machine*
    ```bash
    export CUDA_HOME=<location for CUDA libs> # e.g. /usr/local/cuda
    export CUDNN_HOME=<location for cuDNN libs> # e.g. /usr/local/cuda
    export CUDACXX=<location for NVCC> #e.g. /usr/local/cuda/bin/nvcc
    ```

3. Create the ONNX Runtime Python wheel

   ```bash
   ./build.sh --config=RelWithDebInfo --enable_training --build_wheel --use_cuda --cuda_home {location of cuda libs eg. /usr/local/cuda/} --cudnn_home {location of cudnn libs eg./usr/local/cuda/} --cuda_version={version for eg. 11.8}
   ```

4. Install the .whl file in `./build/Linux/RelWithDebInfo/dist` for ONNX Runtime Training.

    ```bash
    python -m pip install build/Linux/RelWithDebInfo/dist/*.whl
    ```

That's it! Once the build is complete, you should be able to use the ONNX Runtime libraries and executables in your projects. Note that these steps are general and may need to be adjusted based on your specific environment and requirements. For more information, you can ask for help on the [ONNX Runtime GitHub community](https://github.com/microsoft/onnxruntime/discussions/new?category=q-a).

## GPU / ROCm
### Prerequisites
{: .no_toc }

The default AMD GPU build requires ROCm software toolkit installed on the system:

* [ROCm](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4/page/How_to_Install_ROCm.html#_How_to_Install) 5.4

### Build instructions
{: .no_toc }

1. Checkout this code repo with

    ```bash
    git clone https://github.com/microsoft/onnxruntime
    cd onnxruntime
    ```

2. Create the ONNX Runtime Python wheel

   ```bash
   ./build.sh --config Release --enable_training --build_wheel --parallel --skip_tests --use_rocm --rocm_home /opt/rocm
   ```

3. Install the .whl file in `./build/Linux/RelWithDebInfo/dist` for ONNX Runtime Training.

    ```bash
    python -m pip install build/Linux/RelWithDebInfo/dist/*.whl
    ```

## DNNL and MKLML

### Build Instructions
{: .no_toc }
#### Linux

`./build.sh --enable_training --use_dnnl`

#### Windows

`.\build.bat --enable_training --use_dnnl`

Add `--build_wheel` to build the ONNX Runtime wheel.

This will produce a .whl file in `build/Linux/RelWithDebInfo/dist` for ONNX Runtime Training.
