---
title: Build with different EPs
parent: Build ONNX Runtime
description: Learm how to build ONNX Runtime from source for different execution providers
nav_order: 3
redirect_from: /docs/how-to/build/eps 
---

# Build ONNX Runtime with Execution Providers
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Execution Provider Shared Libraries

The oneDNN, TensorRT, OpenVINO™, and CANN providers are built as shared libraries vs being statically linked into the main onnxruntime. This enables them to be loaded only when needed, and if the dependent libraries of the provider are not installed onnxruntime will still run fine, it just will not be able to use that provider. For non shared library providers, all dependencies of the provider must exist to load onnxruntime.

### Built files
{: .no_toc }

On Windows, shared provider libraries will be named 'onnxruntime_providers_\*.dll' (for example onnxruntime_providers_openvino.dll).
On Unix, they will be named 'libonnxruntime_providers_\*.so'
On Mac, they will be named 'libonnxruntime_providers_\*.dylib'.

There is also a shared library that shared providers depend on called onnxruntime_providers_shared (with the same naming convension applied as above).

Note: It is not recommended to put these libraries in a system location or added to a library search path (like LD_LIBRARY_PATH on Unix). If multiple versions of onnxruntime are installed on the system this can make them find the wrong libraries and lead to undefined behavior.

### Loading the shared providers
{: .no_toc }

Shared provider libraries are loaded by the onnxruntime code (do not load or depend on them in your client code). The API for registering shared or non shared providers is identical, the difference is that shared ones will be loaded at runtime when the provider is added to the session options (through a call like OrtSessionOptionsAppendExecutionProvider_OpenVINO or SessionOptionsAppendExecutionProvider_OpenVINO in the C API).
If a shared provider library cannot be loaded (if the file doesn't exist, or its dependencies don't exist or not in the path) then an error will be returned.

The onnxruntime code will look for the provider shared libraries in the same location as the onnxruntime shared library is (or the executable statically linked to the static library version).

---

## CUDA

### Prerequisites
{: .no_toc }

* Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) according to the [version compatibility matrix](../execution-providers/CUDA-ExecutionProvider.md#requirements).
  * The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home` parameter.
  * The path to the cuDNN installation (include the `cuda` folder in the path) must be provided via the cuDNN_PATH environment variable, or `--cudnn_home` parameter. The cuDNN path should contain `bin`, `include` and `lib` directories.
  * The path to the cuDNN bin directory must be added to the PATH environment variable so that cudnn64_8.dll is found.


### Build Instructions
{: .no_toc }

#### Windows

```
.\build.bat --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path>
```

#### Linux

```
./build.sh --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path>
```

A Dockerfile is available [here](https://github.com/microsoft/onnxruntime/blob/main/dockerfiles#cuda).

### Notes on older versions of ONNX Runtime, CUDA and Visual Studio
{: .no_toc }

* Depending on compatibility between the CUDA, cuDNN, and Visual Studio versions you are using, you may need to explicitly install an earlier version of the MSVC toolset.
* For older version of ONNX Runtime and CUDA, and Visual Studio:
    * CUDA 10.0 is [known to work](https://devblogs.microsoft.com/cppblog/cuda-10-is-now-available-with-support-for-the-latest-visual-studio-2017-versions/) with toolsets from 14.11 up to 14.16 (Visual Studio 2017 15.9), and should continue to work with future Visual Studio versions
    * CUDA 9.2 is known to work with the 14.11 MSVC toolset (Visual Studio 15.3 and 15.4)
        * To install the 14.11 MSVC toolset, see [this page](https://blogs.msdn.microsoft.com/vcblog/2017/11/15/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2017).
        * To use the 14.11 toolset with a later version of Visual Studio 2017 you have two options:
          1. Setup the Visual Studio environment variables to point to the 14.11 toolset by running vcvarsall.bat, prior to running the build script. e.g. if you have VS2017 Enterprise, an x64 build would use the following command `"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.11` For convenience, .\build.amd64.1411.bat will do this and can be used in the same way as .\build.bat. e.g. ` .\build.amd64.1411.bat --use_cuda`

          2. Alternatively, if you have CMake 3.13 or later you can specify the toolset version via the `--msvc_toolset` build script parameter. e.g. `.\build.bat --msvc_toolset 14.11`

* If you have multiple versions of CUDA installed on a Windows machine and are building with Visual Studio, CMake will use the build files for the highest version of CUDA it finds in the BuildCustomization folder.
e.g. C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\IDE\VC\VCTargets\BuildCustomizations\.
If you want to build with an earlier version, you must temporarily remove the 'CUDA x.y.*' files for later versions from this directory.

---

## TensorRT

See more information on the TensorRT Execution Provider [here](../execution-providers/TensorRT-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

* Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn)
   * The TensorRT execution provider for ONNX Runtime is built and tested up to CUDA 11.8 and cuDNN 8.9. Check [here](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements) for more version information.
   * The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home` parameter. The CUDA path should contain `bin`, `include` and `lib` directories.
   * The path to the CUDA `bin` directory must be added to the PATH environment variable so that `nvcc` is found.
   * The path to the cuDNN installation (path to cudnn bin/include/lib) must be provided via the cuDNN_PATH environment variable, or `--cudnn_home` parameter.
     * On Windows, cuDNN requires [zlibwapi.dll](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows). Feel free to place this dll under `path_to_cudnn/bin`  
 * Follow [instructions for installing TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
   * The TensorRT execution provider for ONNX Runtime is built and tested up to TensorRT 8.6.
   * The path to TensorRT installation must be provided via the `--tensorrt_home` parameter.
   * ONNX Runtime uses TensorRT built-in parser from `tensorrt_home` by default.
   * To use open-sourced [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt/tree/main) parser instead, add `--use_tensorrt_oss_parser` parameter in build commands below.
       * The default version of open-sourced onnx-tensorrt parser is encoded in [cmake/deps.txt](https://github.com/microsoft/onnxruntime/blob/main/cmake/deps.txt).
       * To specify a different version of onnx-tensorrt parser:
         * Select the commit of [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt/commits) that you preferred;
         * Run `sha1sum` command with downloaded onnx-tensorrt zip file to acquire the SHA1 hash 
         * Update [cmake/deps.txt](https://github.com/microsoft/onnxruntime/blob/main/cmake/deps.txt) with updated onnx-tensorrt commit and hash info.

### Build Instructions
{: .no_toc }

#### Windows
```bash
# to build with tensorrt built-in parser
.\build.bat --cudnn_home <path to cuDNN home> --cuda_home <path to CUDA home> --use_tensorrt --tensorrt_home <path to TensorRT home> --cmake_generator "Visual Studio 17 2022"

# to build with specific version of open-sourced onnx-tensorrt parser configured in cmake/deps.txt
.\build.bat --cudnn_home <path to cuDNN home> --cuda_home <path to CUDA home> --use_tensorrt --tensorrt_home <path to TensorRT home> --use_tensorrt_oss_parser --cmake_generator "Visual Studio 17 2022" 
```

#### Linux

```bash
# to build with tensorrt built-in parser
./build.sh --cudnn_home <path to cuDNN e.g. /usr/lib/x86_64-linux-gnu/> --cuda_home <path to folder for CUDA e.g. /usr/local/cuda> --use_tensorrt --tensorrt_home <path to TensorRT home>

# to build with specific version of open-sourced onnx-tensorrt parser configured in cmake/deps.txt
./build.sh  --cudnn_home <path to cuDNN e.g. /usr/lib/x86_64-linux-gnu/> --cuda_home <path to folder for CUDA e.g. /usr/local/cuda> --use_tensorrt --use_tensorrt_oss_parser --tensorrt_home <path to TensorRT home> --skip_submodule_sync
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/tree/main/dockerfiles#tensorrt)

---

## NVIDIA Jetson TX1/TX2/Nano/Xavier

### Build Instructions
{: .no_toc }

These instructions are for the latest [JetPack SDK 5.1.2](https://developer.nvidia.com/embedded/jetpack-sdk-512).

1. Clone the ONNX Runtime repo on the Jetson host

   ```bash
   git clone --recursive https://github.com/microsoft/onnxruntime
   ```

2. Specify the CUDA compiler, or add its location to the PATH.

   1. Starting with **CUDA 11.8**, Jetson users on **JetPack 5.0+** can upgrade to the latest CUDA release without updating the JetPack version or Jetson Linux BSP (Board Support Package). CUDA version 11.8 with JetPack 5.1.2 has been tested on Jetson when building ONNX Runtime 1.16. 

      1. Check [this official blog](https://developer.nvidia.com/blog/simplifying-cuda-upgrades-for-nvidia-jetson-users/) for CUDA 11.8 upgrade instruction.

      2. CUDA 12.x is only available to Jetson Orin and newer series (CUDA compute capability >= 8.7). Check [here](https://developer.nvidia.com/cuda-gpus#collapse5) for compute capability datasheet.

   2. CMake can't automatically find the correct `nvcc` if it's not in the `PATH`. `nvcc` can be added to `PATH` via:

      ```bash
      export PATH="/usr/local/cuda/bin:${PATH}"
      ```

       or:

      ```bash
      export CUDACXX="/usr/local/cuda/bin/nvcc"
      ```

3. Install the ONNX Runtime build dependencies on the Jetpack 5.1.2 host:

   ```bash
   sudo apt install -y --no-install-recommends \
     build-essential software-properties-common libopenblas-dev \
     libpython3.8-dev python3-pip python3-dev python3-setuptools python3-wheel
   ```

4. Cmake is needed to build ONNX Runtime. For ONNX Runtime 1.16, the minimum required CMake version is 3.26 (version 3.27.4 has been tested). This can be either installed by:

   1. (Unix/Linux) Build from source. Download sources from [https://cmake.org/download/](https://cmake.org/download/)
      and follow [https://cmake.org/install/](https://cmake.org/install/) to build from source. 
   2. (Ubuntu) Install deb package via apt repository: e.g [https://apt.kitware.com/](https://apt.kitware.com/)

5. Build the ONNX Runtime Python wheel (update path to CUDA/CUDNN/TensorRT libraries if necessary):

   1. Build `onnxruntime-gpu` wheel with CUDA and TensorRT support:

      ```bash
      ./build.sh --config Release --update --build --parallel --build_wheel \
      --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
      --tensorrt_home /usr/lib/aarch64-linux-gnu
      ```

​	Notes:

* By default, `onnxruntime-gpu` wheel file will be captured under `path_to/onnxruntime/build/Linux/Release/dist/` (build path can be customized by adding `--build_dir` followed by a customized path to the build command above).

* For a portion of Jetson devices like the Xavier series, higher power mode involves more cores (up to 6) to compute but it consumes more resources when building ONNX Runtime. Update `--parallel` with smaller value if system resource is limited and OOM happens. 
## oneDNN

See more information on oneDNN (formerly DNNL) [here](../execution-providers/oneDNN-ExecutionProvider.md).

### Build Instructions
{: .no_toc }


The DNNL execution provider can be built for Intel CPU or GPU. To build for Intel GPU, install [Intel SDK for OpenCL Applications](https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html) or build OpenCL from [Khronos OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK). Pass in the OpenCL SDK path as dnnl_opencl_root to the build command. Install the latest GPU driver - [Windows graphics driver](https://downloadcenter.intel.com/product/80939/Graphics), [Linux graphics compute runtime and OpenCL driver](https://github.com/intel/compute-runtime/releases).

For CPU
#### Windows
`.\build.bat --use_dnnl`

#### Linux
`./build.sh --use_dnnl`

For GPU
#### Windows 

`.\build.bat --use_dnnl --dnnl_gpu_runtime ocl --dnnl_opencl_root "c:\program files (x86)\intelswtools\sw_dev_tools\opencl\sdk"`
#### Linux

`./build.sh --use_dnnl --dnnl_gpu_runtime ocl --dnnl_opencl_root "/opt/intel/sw_dev_tools/opencl-sdk"`

#### Build Phython Wheel


OneDNN EP build supports building Python wheel for both Windows and linux using flag --build_wheel

`.\build.bat --config RelWithDebInfo --parallel --build_shared_lib --cmake_generator "Visual Studio 16 2019" --build_wheel --use_dnnl --dnnl_gpu_runtime ocl --dnnl_opencl_root "C:\Program Files (x86)\IntelSWTools\system_studio_2020\OpenCL\sdk"`

---

## OpenVINO

See more information on the OpenVINO™ Execution Provider [here](../execution-providers/OpenVINO-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

1. Install the OpenVINO™ offline/online installer from Intel<sup>®</sup> Distribution of OpenVINO™<sup>TM</sup> Toolkit **Release 2023.1** for the appropriate OS and target hardware:
   * [Windows - CPU, GPU](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?VERSION=v_2023_1_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE).
   * [Linux - CPU, GPU](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?VERSION=v_2023_1_0&OP_SYSTEM=LINUX&DISTRIBUTION=ARCHIVE)

   Follow [documentation](https://docs.openvino.ai/2023.1/index.html) for detailed instructions.

  *2023.1 is the recommended OpenVINO™ version. [OpenVINO™ 2022.1](https://docs.openvino.ai/archive/2022.1/index.html) is minimal OpenVINO™ version requirement.*
  *The minimum ubuntu version to support 2023.1 is 18.04.*

2. Configure the target hardware with specific follow on instructions:
   * To configure Intel<sup>®</sup> Processor Graphics(GPU) please follow these instructions: [Windows](https://docs.openvino.ai/latest/openvino_docs_install_guides_configurations_for_intel_gpu.html#gpu-guide-windows), [Linux](https://docs.openvino.ai/latest/openvino_docs_install_guides_configurations_for_intel_gpu.html#linux)


3. Initialize the OpenVINO™ environment by running the setupvars script as shown below. This is a required step:
   * For Windows:
   ```
      C:\<openvino_install_directory>\setupvars.bat
   ```
   * For Linux:
   ```
      $ source <openvino_install_directory>/setupvars.sh
   ```
   **Note:** If you are using a dockerfile to use OpenVINO™ Execution Provider, sourcing OpenVINO™ won't be possible within the dockerfile. You would have to explicitly set the LD_LIBRARY_PATH to point to OpenVINO™ libraries location. Refer our [dockerfile](https://github.com/microsoft/onnxruntime/blob/main/dockerfiles/Dockerfile.openvino).

### Build Instructions
{: .no_toc }

#### Windows

```
.\build.bat --config RelWithDebInfo --use_openvino <hardware_option> --build_shared_lib --build_wheel
```

*Note: The default Windows CMake Generator is Visual Studio 2019, but you can also use the newer Visual Studio 2022 by passing `--cmake_generator "Visual Studio 17 2022"` to `.\build.bat`*

#### Linux

```bash
./build.sh --config RelWithDebInfo --use_openvino <hardware_option> --build_shared_lib --build_wheel
```

* `--build_wheel` Creates python wheel file in dist/ folder. Enable it when building from source and/or while building with CXX11_ABI=1 of OpenVINO.
* `--use_openvino` builds the OpenVINO™ Execution Provider in ONNX Runtime.
* `<hardware_option>`: Specifies the default hardware target for building OpenVINO™ Execution Provider. This can be overriden dynamically at runtime with another option (refer to [OpenVINO™-ExecutionProvider](../execution-providers/OpenVINO-ExecutionProvider.md#summary-of-options) for more details on dynamic device selection). Below are the options for different Intel target devices.

Refer to [Intel GPU device naming convention](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_supported_plugins_GPU.html#device-naming-convention) for specifying the correct hardware target in cases where both integrated and discrete GPU's co-exist.

| Hardware Option | Target Device |
| --------------- | ------------------------|
| <code>CPU_FP32</code> | Intel<sup>®</sup> CPUs |
| <code>GPU_FP32</code> | Intel<sup>®</sup> Integrated Graphics |
| <code>GPU_FP16</code> | Intel<sup>®</sup> Integrated Graphics with FP16 quantization of models |
| <code>GPU.0_FP32</code> | Intel<sup>®</sup> Integrated Graphics |
| <code>GPU.0_FP16</code> | Intel<sup>®</sup> Integrated Graphics with FP16 quantization of models |
| <code>GPU.1_FP32</code> | Intel<sup>®</sup> Discrete Graphics |
| <code>GPU.1_FP16</code> | Intel<sup>®</sup> Discrete Graphics with FP16 quantization of models |
| <code>HETERO:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...</code> | All Intel<sup>®</sup> silicons mentioned above |
| <code>MULTI:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...</code> | All Intel<sup>®</sup> silicons mentioned above |
| <code>AUTO:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...</code> | All Intel<sup>®</sup> silicons mentioned above |

Specifying Hardware Target for HETERO or Multi or AUTO device Build:

HETERO:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...
The DEVICE_TYPE can be any of these devices from this list ['CPU','GPU']

A minimum of two device's should be specified for a valid HETERO or MULTI or AUTO device build.

```
Example's: HETERO:GPU,CPU or AUTO:GPU,CPU or MULTI:GPU,CPU
```

#### Disable subgraph partition Feature
* Builds the OpenVINO™ Execution Provider in ONNX Runtime with sub graph partitioning disabled.

* With this option enabled. Fully supported models run on OpenVINO Execution Provider else they completely fall back to default CPU EP.

* To enable this feature during build time. Use `--use_openvino ` `<hardware_option>_NO_PARTITION`

```
Usage: --use_openvino CPU_FP32_NO_PARTITION or --use_openvino GPU_FP32_NO_PARTITION or
       --use_openvino GPU_FP16_NO_PARTITION 
```

For more information on OpenVINO™ Execution Provider&#39;s ONNX Layer support, Topology support, and Intel hardware enabled, please refer to the document [OpenVINO™-ExecutionProvider](../execution-providers/OpenVINO-ExecutionProvider.md)

---

## QNN
See more information on the QNN execution provider [here](../execution-providers/QNN-ExecutionProvider.md).

### Prerequisites
{: .no_toc }
* Qualcomm AI Engine Direct SDK (Qualcomm Neural Network SDK) [Linux/Android/Windows](https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct)

### Build Instructions
{: .no_toc }

#### Windows (arm64 native build)
```
build.bat --arm64 --use_qnn --qnn_home=[QNN_SDK path] --build_shared_lib --cmake_generator "Visual Studio 17 2022" --skip_submodule_sync --config Release --build_dir \build\Windows
```

build python bindings 
```
build.bat --arm64 --use_qnn --qnn_home=[QNN_SDK path] --build_wheel --cmake_generator "Visual Studio 17 2022" --skip_submodule_sync --config Release --build_dir \build\Windows
```
#### Linux (x64)
```
build.py --use_qnn --qnn_home=[QNN_SDK path] --build_shared_lib --skip_submodule_sync --config Release
```
#### Android (Cross-Compile):

Please reference [Build OnnxRuntime For Android](android.md)
```
# on Windows
build.bat --build_shared_lib --skip_submodule_sync --android --config Release --use_qnn --qnn_home [QNN_SDK path] --android_sdk_path [android_SDK path] --android_ndk_path [android_NDK path] --android_abi arm64-v8a --android_api [api-version] --cmake_generator Ninja --build_dir build\Android

# on Linux
build.sh --build_shared_lib --skip_submodule_sync --android --config Release --use_qnn --qnn_home [QNN_SDK path] --android_sdk_path [android_SDK path] --android_ndk_path [android_NDK path] --android_abi arm64-v8a --android_api [api-version] --cmake_generator Ninja --build_dir build/Android

```

---

## DirectML
See more information on the DirectML execution provider [here](../execution-providers/DirectML-ExecutionProvider.md).
### Windows
{: .no_toc }

```
.\build.bat --use_dml
```
### Notes
{: .no_toc }

The DirectML execution provider supports building for both x64 and x86 architectures. DirectML is only supported on Windows.

---

## ARM Compute Library
See more information on the ACL Execution Provider [here](../execution-providers/community-maintained/ACL-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

* Supported backend: i.MX8QM Armv8 CPUs
* Supported BSP: i.MX8QM BSP
  * Install i.MX8QM BSP: `source fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4*.sh`
* Set up the build environment
```
source /opt/fsl-imx-xwayland/4.*/environment-setup-aarch64-poky-linux
alias cmake="/usr/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
```
* See [Build ARM](inferencing.md#arm) below for information on building for ARM devices

### Build Instructions
{: .no_toc }

1. Configure ONNX Runtime with ACL support:
```
cmake ../onnxruntime-arm-upstream/cmake -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc -Donnxruntime_RUN_ONNX_TESTS=OFF -Donnxruntime_GENERATE_TEST_REPORTS=ON -Donnxruntime_DEV_MODE=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Donnxruntime_USE_CUDA=OFF -Donnxruntime_USE_NSYNC=OFF -Donnxruntime_CUDNN_HOME= -Donnxruntime_USE_JEMALLOC=OFF -Donnxruntime_ENABLE_PYTHON=OFF -Donnxruntime_BUILD_CSHARP=OFF -Donnxruntime_BUILD_SHARED_LIB=ON -Donnxruntime_USE_EIGEN_FOR_BLAS=ON -Donnxruntime_USE_OPENBLAS=OFF -Donnxruntime_USE_ACL=ON -Donnxruntime_USE_DNNL=OFF -Donnxruntime_USE_MKLML=OFF -Donnxruntime_USE_OPENMP=ON -Donnxruntime_USE_TVM=OFF -Donnxruntime_USE_LLVM=OFF -Donnxruntime_ENABLE_MICROSOFT_INTERNAL=OFF -Donnxruntime_USE_BRAINSLICE=OFF -Donnxruntime_USE_EIGEN_THREADPOOL=OFF -Donnxruntime_BUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
The ```-Donnxruntime_USE_ACL=ON``` option will use, by default, the 19.05 version of the Arm Compute Library. To set the right version you can use:
```-Donnxruntime_USE_ACL_1902=ON```, ```-Donnxruntime_USE_ACL_1905=ON```, ```-Donnxruntime_USE_ACL_1908=ON``` or ```-Donnxruntime_USE_ACL_2002=ON```;

To use a library outside the normal environment you can set a custom path by using ```-Donnxruntime_ACL_HOME``` and ```-Donnxruntime_ACL_LIBS``` tags that defines the path to the ComputeLibrary directory and the build directory respectively.

```-Donnxruntime_ACL_HOME=/path/to/ComputeLibrary```, ```-Donnxruntime_ACL_LIBS=/path/to/build```


2. Build ONNX Runtime library, test and performance application:
```
make -j 6
```

3. Deploy ONNX runtime on the i.MX 8QM board
```
libonnxruntime.so.0.5.0
onnxruntime_perf_test
onnxruntime_test_all
```

### Native Build Instructions 
{: .no_toc }

*Validated on Jetson Nano and Jetson Xavier*

1. Build ACL Library (skip if already built)

    ```bash
    cd ~
    git clone -b v20.02 https://github.com/Arm-software/ComputeLibrary.git
    cd ComputeLibrary
    sudo apt-get install -y scons g++-arm-linux-gnueabihf
    scons -j8 arch=arm64-v8a  Werror=1 debug=0 asserts=0 neon=1 opencl=1 examples=1 build=native
    ```

1. Cmake is needed to build ONNX Runtime. Because the minimum required version is 3.13,
   it is necessary to build CMake from source. Download Unix/Linux sources from https://cmake.org/download/
   and follow https://cmake.org/install/ to build from source. Version 3.17.5 and 3.18.4 have been tested on Jetson.

1. Build onnxruntime with --use_acl flag with one of the supported ACL version flags. (ACL_1902 | ACL_1905 | ACL_1908 | ACL_2002)

---

## ArmNN

See more information on the ArmNN Execution Provider [here](../execution-providers/community-maintained/ArmNN-ExecutionProvider.md).

### Prerequisites
{: .no_toc }


* Supported backend: i.MX8QM Armv8 CPUs
* Supported BSP: i.MX8QM BSP
  * Install i.MX8QM BSP: `source fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4*.sh`
* Set up the build environment

```bash
source /opt/fsl-imx-xwayland/4.*/environment-setup-aarch64-poky-linux
alias cmake="/usr/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
```

* See [Build ARM](inferencing.md#arm) below for information on building for ARM devices

### Build Instructions
{: .no_toc }


```bash
./build.sh --use_armnn
```

The Relu operator is set by default to use the CPU execution provider for better performance. To use the ArmNN implementation build with --armnn_relu flag

```bash
./build.sh --use_armnn --armnn_relu
```

The Batch Normalization operator is set by default to use the CPU execution provider. To use the ArmNN implementation build with --armnn_bn flag

```bash
./build.sh --use_armnn --armnn_bn
```

To use a library outside the normal environment you can set a custom path by providing the --armnn_home and --armnn_libs parameters to define the path to the ArmNN home directory and build directory respectively. 
The ARM Compute Library home directory and build directory must also be available, and can be specified if needed using --acl_home and --acl_libs respectively.

```bash
./build.sh --use_armnn --armnn_home /path/to/armnn --armnn_libs /path/to/armnn/build  --acl_home /path/to/ComputeLibrary --acl_libs /path/to/acl/build
```

---

## RKNPU
See more information on the RKNPU Execution Provider [here](../execution-providers/community-maintained/RKNPU-ExecutionProvider.md).

### Prerequisites
{: .no_toc }


* Supported platform: RK1808 Linux
* See [Build ARM](inferencing.md#arm) below for information on building for ARM devices
* Use gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu instead of gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf, and modify CMAKE_CXX_COMPILER & CMAKE_C_COMPILER in tool.cmake:
  
```
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
```

### Build Instructions
{: .no_toc }

#### Linux

1. Download [rknpu_ddk](https://github.com/airockchip/rknpu_ddk.git) to any directory.

2. Build ONNX Runtime library and test:

    ```bash
    ./build.sh --arm --use_rknpu --parallel --build_shared_lib --build_dir build_arm --config MinSizeRel --cmake_extra_defines RKNPU_DDK_PATH=<Path To rknpu_ddk> CMAKE_TOOLCHAIN_FILE=<Path To tool.cmake> ONNX_CUSTOM_PROTOC_EXECUTABLE=<Path To protoc>
    ```

3. Deploy ONNX runtime and librknpu_ddk.so on the RK1808 board:

    ```bash
    libonnxruntime.so.1.2.0
    onnxruntime_test_all
    rknpu_ddk/lib64/librknpu_ddk.so
    ```

---

## AMD Vitis AI
See more information on the Vitis AI Execution Provider [here](../execution-providers/Vitis-AI-ExecutionProvider.md).

### Windows
{: .no_toc }

From the Visual Studio Developer Command Prompt or Developer PowerShell, execute the following command:

```
.\build.bat --use_vitisai --build_shared_lib --parallel --config Release
```

If you wish to leverage the Python APIs, please include the `--build_wheel` flag:

```
.\build.bat --use_vitisai --build_shared_lib --parallel --config Release --build_wheel
```

You can override also override the installation location by specifying CMAKE_INSTALL_PREFIX via the cmake_extra_defines parameter.
e.g.

```
.\build.bat --use_vitisai --build_shared_lib --parallel --config Release --cmake_extra_defines CMAKE_INSTALL_PREFIX=D:\onnxruntime
```
### Linux
{: .no_toc }

Currently Linux support is only enabled for AMD Adapable SoCs.  Please refer to the guidance [here](../execution-providers/Vitis-AI-ExecutionProvider.md#amd-adaptable-soc-installation) for SoC targets.

---

## AMD MIGraphX

See more information on the MIGraphX Execution Provider [here](../execution-providers/MIGraphX-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

* Install [ROCm](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4/page/How_to_Install_ROCm.html#_How_to_Install)
  * The MIGraphX execution provider for ONNX Runtime is built and tested with ROCm5.4
* Install [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX)
  * The path to MIGraphX installation must be provided via the `--migraphx_home parameter`.

### Build Instructions
{: .no_toc }

#### Linux

```bash
./build.sh --config <Release|Debug|RelWithDebInfo> --use_migraphx --migraphx_home <path to MIGraphX home>
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/blob/main/dockerfiles#migraphx).

## AMD ROCm

See more information on the ROCm Execution Provider [here](../execution-providers/ROCm-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

* Install [ROCm](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4/page/How_to_Install_ROCm.html#_How_to_Install)
  * The ROCm execution provider for ONNX Runtime is built and tested with ROCm5.4

### Build Instructions
{: .no_toc }

#### Linux

```bash
./build.sh --config <Release|Debug|RelWithDebInfo> --use_rocm --rocm_home <path to ROCm home>
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/tree/main/dockerfiles#rocm).

## NNAPI

Usage of NNAPI on Android platforms is via the NNAPI Execution Provider (EP).

See the [NNAPI Execution Provider](../execution-providers/NNAPI-ExecutionProvider.md) documentation for more details.

The pre-built ONNX Runtime Mobile package for Android includes the NNAPI EP.

If performing a custom build of ONNX Runtime, support for the NNAPI EP or CoreML EP must be enabled when building.

### Create a minimal build with NNAPI EP support

Please see [the instructions](./android.md) for setting up the Android environment required to build. The Android build can be cross-compiled on Windows or Linux.

Once you have all the necessary components setup, follow the instructions to [create the custom build](./custom.md), with the following changes:

* Replace `--minimal_build` with `--minimal_build extended` to enable support for execution providers that dynamically create kernels at runtime, which is required by the NNAPI EP.
* Add `--use_nnapi` to include the NNAPI EP in the build

#### Example build commands with the NNAPI EP enabled

Windows example:

```dos
<ONNX Runtime repository root>.\build.bat --config MinSizeRel --android --android_sdk_path D:\Android --android_ndk_path D:\Android\ndk\21.1.6352462\ --android_abi arm64-v8a --android_api 29 --cmake_generator Ninja --minimal_build extended --use_nnapi --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>
```

Linux example:

```bash
<ONNX Runtime repository root>./build.sh --config MinSizeRel --android --android_sdk_path /Android --android_ndk_path /Android/ndk/21.1.6352462/ --android_abi arm64-v8a --android_api 29 --minimal_build extended --use_nnapi --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>`
```

## CoreML

Usage of CoreML on iOS and macOS platforms is via the CoreML EP.

See the [CoreML Execution Provider](../execution-providers/CoreML-ExecutionProvider.md) documentation for more details.

The pre-built ONNX Runtime Mobile package for iOS includes the CoreML EP.

### Create a minimal build with CoreML EP support

Please see [the instructions](./ios.md) for setting up the iOS environment required to build. The iOS/macOS build must be performed on a mac machine.

Once you have all the necessary components setup, follow the instructions to [create the custom build](./custom.md), with the following changes:

* Replace `--minimal_build` with `--minimal_build extended` to enable support for execution providers that dynamically create kernels at runtime, which is required by the CoreML EP.
* Add `--use_coreml` to include the CoreML EP in the build

## XNNPACK

Usage of XNNPACK on Android/iOS/Windows/Linux platforms is via the XNNPACK EP.

See the [XNNPACK Execution Provider](../execution-providers/Xnnpack-ExecutionProvider.md) documentation for more details.

The pre-built ONNX Runtime package([`onnxruntime-android`](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android)) for Android includes the XNNPACK EP.

The pre-built ONNX Runtime Mobile package for iOS, `onnxruntime-c` and `onnxruntime-objc` in [CocoaPods](https://cocoapods.org/), includes the XNNPACK EP. (Package `onnxruntime-objc` with XNNPACK will be available since 1.14.)


If performing a custom build of ONNX Runtime, support for the XNNPACK EP must be enabled when building.

### Build for Android
#### Create a minimal build with XNNPACK EP support

Please see [the instructions](./android.md) for setting up the Android environment required to build. The Android build can be cross-compiled on Windows or Linux.

Once you have all the necessary components setup, follow the instructions to [create the custom build](./custom.md), with the following changes:

* Replace `--minimal_build` with `--minimal_build extended` to enable support for execution providers that dynamically create kernels at runtime, which is required by the XNNPACK EP.
* Add `--use_xnnpack` to include the XNNPACK EP in the build

##### Example build commands with the XNNPACK EP enabled

Windows example:

```bash
<ONNX Runtime repository root>.\build.bat --config MinSizeRel --android --android_sdk_path D:\Android --android_ndk_path D:\Android\ndk\21.1.6352462\ --android_abi arm64-v8a --android_api 29 --cmake_generator Ninja --minimal_build extended --use_xnnpack --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>
```

Linux example:

```bash
<ONNX Runtime repository root>./build.sh --config MinSizeRel --android --android_sdk_path /Android --android_ndk_path /Android/ndk/21.1.6352462/ --android_abi arm64-v8a --android_api 29 --minimal_build extended --use_xnnpack --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>`
```
If you don't mind MINIMAL build, you can use the following command to build XNNPACK EP for Android:
Linux example:
```bash
./build.sh --cmake_generator "Ninja" --android  --android_sdk_path /Android --android_ndk_path /Android/ndk/21.1.6352462/ --android_abi arm64-v8a --android_api 29 --use_xnnpack
```
### Build for iOS (available since 1.14)
A Mac machine is required to build package for iOS. Please follow this [guide](./ios.md) to set up environment firstly.
#### Create a minimal build with XNNPACK EP support

Once you have all the necessary components setup, follow the instructions to [create the custom build](./custom.md), with the following changes:

* Replace `--minimal_build` with `--minimal_build extended` to enable support for execution providers that dynamically create kernels at runtime, which is required by the XNNPACK EP.
* Add `--use_xnnpack` to include the XNNPACK EP in the build

```dos
<ONNX Runtime repository root>./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --ios_sysroot iphoneos --osx_arch arm64 --apple_deploy_target <minimal iOS version> --use_xnnpack --minimal_build extended --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>
```

### Build for Windows
```dos
<ONNX Runtime repository root>.\build.bat --config <Release|Debug|RelWithDebInfo> --use_xnnpack
```
### Build for Linux
```bash
<ONNX Runtime repository root>./build.sh --config <Release|Debug|RelWithDebInfo> --use_xnnpack
```

---

## CANN

See more information on the CANN Execution Provider [here](../execution-providers/community-maintained/CANN-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

1. Install the CANN Toolkit for the appropriate OS and target hardware by following [documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/51RC1alphaX/softwareinstall/instg/atlasdeploy_03_0017.html) for detailed instructions, please.

2. Initialize the CANN environment by running the script as shown below.

   ```bash
   # Default path, change it if needed.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

### Build Instructions
{: .no_toc }

#### Linux

```bash
./build.sh --config <Release|Debug|RelWithDebInfo> --build_shared_lib --parallel --use_cann
```

### Notes
{: .no_toc }

* The CANN execution provider supports building for both x64 and aarch64 architectures.
* CANN excution provider now is only supported on Linux.

## Azure

See the [Azure Execution Provider](../execution-providers/Azure-ExecutionProvider.md) documentation for more details.

### Prerequisites

For Linux, before building, please:

* install openssl dev package into the system, which is openssl-dev for redhat and libssl-dev for ubuntu.
* if have multiple openssl dev versions installed in the system, please set environment variable "OPENSSL_ROOT_DIR" to the desired version, for example:

```base
set OPENSSL_ROOT_DIR=/usr/local/ssl3.x/
```

### Build Instructions

#### Windows

```dos
build.bat --config <Release|Debug|RelWithDebInfo> --build_shared_lib --build_wheel --use_azure
```

#### Linux

```bash
./build.sh --config <Release|Debug|RelWithDebInfo> --build_shared_lib --build_wheel --use_azure
```
