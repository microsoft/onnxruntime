---
title: Build from source
parent: How to
nav_order: 5
---

# Build ONNX Runtime from source
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Inference

### CPU

#### Prerequisites
* Checkout the source tree:
   ```
   git clone --recursive https://github.com/Microsoft/onnxruntime
   cd onnxruntime
   ```
* Install cmake-3.13 or higher from https://cmake.org/download/.


#### Build Instructions

##### Windows

Open Developer Command Prompt for Visual Studio version you are going to use. This will properly setup the environment including paths to your compiler, linker, utilities and header files.
```
.\build.bat --config RelWithDebInfo --build_shared_lib --parallel
```
The default Windows CMake Generator is Visual Studio 2017, but you can also use the newer Visual Studio 2019 by passing `--cmake_generator "Visual Studio 16 2019"` to `.\build.bat`

##### Linux

```
./build.sh --config RelWithDebInfo --build_shared_lib --parallel
```

##### macOS

By default, ORT is configured to be built for a minimum target macOS version of 10.12.
The shared library in the release Nuget(s) and the Python wheel may be installed on macOS versions of 10.12+.

If you would like to use [Xcode](https://developer.apple.com/xcode/) to build the onnxruntime for x86_64 macOS, please add the --user_xcode argument in the command line.

Without this flag, the cmake build generator will be Unix makefile by default.
Also, if you want to cross-compile for Apple Silicon in an Intel-based MacOS machine, please add the argument --osx_arch arm64 with cmake > 3.19. Note: unit tests will be skipped due to the incompatible CPU instruction set.

##### Notes

* Please note that these instructions build the debug build, which may have performance tradeoffs
* To build the version from each release (which include Windows, Linux, and Mac variants), see these .yml files for reference: [CPU](https://github.com/microsoft/onnxruntime/blob/master/tools/ci_build/github/azure-pipelines/nuget/cpu-esrp-pipeline.yml), [GPU](https://github.com/microsoft/onnxruntime/blob/master/tools/ci_build/github/azure-pipelines/nuget/gpu-esrp-pipeline.yml)
* The build script runs all unit tests by default (for native builds and skips tests by default for cross-compiled builds).
* If you need to install protobuf 3.6.1 from source code (cmake/external/protobuf), please note:
   * CMake flag protobuf\_BUILD\_SHARED\_LIBS must be turned OFF. After the installation, you should have the 'protoc' executable in your PATH. It is recommended to run `ldconfig` to make sure protobuf libraries are found.
   * If you installed your protobuf in a non standard location it would be helpful to set the following env var:`export CMAKE_ARGS="-DONNX_CUSTOM_PROTOC_EXECUTABLE=full path to protoc"` so the ONNX build can find it. Also run `ldconfig <protobuf lib folder path>` so the linker can find protobuf libraries.
* If you'd like to install onnx from source code (cmake/external/onnx), use:
    ```
    export ONNX_ML=1
    python3 setup.py bdist_wheel
    pip3 install --upgrade dist/*.whl
    ```
---

### Supported architectures and build environments

#### Architectures

|           | x86_32       | x86_64       | ARM32v7      | ARM64        |
|-----------|:------------:|:------------:|:------------:|:------------:|
|Windows    | YES          | YES          |  YES         | YES          |
|Linux      | YES          | YES          |  YES         | YES          |
|macOS      | NO           | YES          |  NO          | NO           |

#### Environments

| OS          | Supports CPU | Supports GPU| Notes                              |
|-------------|:------------:|:------------:|------------------------------------|
|Windows 10   | YES          | YES         | VS2019 through the latest VS2015 are supported |
|Windows 10 <br/> Subsystem for Linux | YES         | NO        |         |
|Ubuntu 16.x  | YES          | YES         | Also supported on ARM32v7 (experimental) |
|macOS        | YES          | NO         |    |

GCC 4.x and below are not supported.

#### OS/Compiler Matrix

| OS/Compiler | Supports VC  | Supports GCC     |  Supports Clang  |
|-------------|:------------:|:----------------:|:----------------:|
|Windows 10   | YES          | Not tested       | Not tested       |
|Linux        | NO           | YES(gcc>=4.8)    | Not tested       |
|macOS        | NO           | Not tested       | YES (Minimum version required not ascertained)|

---

#### Common Build Instructions

|Description|Command|Additional details|
|-----------|-----------|-----------|
|**Basic build**|build.bat (Windows)<br>./build.sh (Linux)||
|**Release build**|--config Release|Release build. Other valid config values are RelWithDebInfo and Debug.|
|**Use OpenMP**|--use_openmp|OpenMP will parallelize some of the code for potential performance improvements. This is not recommended for running on single threads.|
|**Build using parallel processing**|--parallel|This is strongly recommended to speed up the build.|
|**Build Shared Library**|--build_shared_lib||
|**Enable Training support**|--enable_training||

#### APIs and Language Bindings

|API|Command|Additional details|
|-----------|-----------|-----------|
|**Python**|--build_wheel||
|**C# and C packages**|--build_nuget|Builds C# bindings and creates nuget package. Currently supported on Windows and Linux only. Implies `--build_shared_lib` <br> Detailed instructions can be found [below](#build-nuget-packages).|
|**WindowsML**|--use_winml<br>--use_dml<br>--build_shared_lib|WindowsML depends on DirectML and the OnnxRuntime shared library|
|**Java**|--build_java|Creates an onnxruntime4j.jar in the build directory, implies `--build_shared_lib`<br>Compiling the Java API requires [gradle](https://gradle.org) v6.1+ to be installed in addition to the usual requirements.|
|**Node.js**|--build_nodejs|Build Node.js binding. Implies `--build_shared_lib`|

---
### Reduced Operator Kernel Build
Reduced Operator Kernel builds allow you to customize the kernels in the build to provide smaller binary sizes - [see instructions](https://github.com/microsoft/onnxruntime/blob/master/docs/Reduced_Operator_Kernel_build.md).

### ONNX Runtime for Mobile Platforms
For builds compatible with mobile platforms, see more details in [ONNX_Runtime_for_Mobile_Platforms.md](https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_for_Mobile_Platforms.md). Android and iOS build instructions can be found below on this page - [Android](#android), [iOS](#ios)

### Build ONNX Runtime Server on Linux
Read more about ONNX Runtime Server [here](https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Server_Usage.md).

Build instructions are [here](https://github.com/microsoft/onnxruntime/blob/master/docs/Server.md)

### Build Nuget packages

Currently only supported on Windows and Linux.

#### Prerequisites

* dotnet is required for building csharp bindings and creating managed nuget package. Follow the instructions [here](https://dotnet.microsoft.com/download) to download dotnet. Tested with versions 2.1 and 3.1.
* nuget.exe. Follow the instructions [here](https://docs.microsoft.com/en-us/nuget/install-nuget-client-tools#nugetexe-cli) to download nuget
  * On Windows, downloading nuget is straightforward and simply following the instructions above should work.
  * On Linux, nuget relies on Mono runtime and therefore this needs to be setup too. Above link has all the information to setup Mono and nuget. The instructions can directly be found [here](https://www.mono-project.com/docs/getting-started/install/). In some cases it is required to run `sudo apt-get install mono-complete` after installing mono.

#### Build Instructions
##### Windows
```
.\build.bat --build_nuget
```

##### Linux
```
./build.sh --build_nuget
```
Nuget packages are created under <native_build_dir>\nuget-artifacts

---

### Execution Provider Shared Libraries

The DNNL, TensorRT, and OpenVINO providers are built as shared libraries vs being statically linked into the main onnxruntime. This enables them to be loaded only when needed, and if the dependent libraries of the provider are not installed onnxruntime will still run fine, it just will not be able to use that provider. For non shared library providers, all dependencies of the provider must exist to load onnxruntime.

#### Built files

On Windows, shared provider libraries will be named 'onnxruntime_providers_\*.dll' (for example onnxruntime_providers_openvino.dll).
On Unix, they will be named 'libonnxruntime_providers_\*.so'
On Mac, they will be named 'libonnxruntime_providers_\*.dylib'.

There is also a shared library that shared providers depend on called onnxruntime_providers_shared (with the same naming convension applied as above).

Note: It is not recommended to put these libraries in a system location or added to a library search path (like LD_LIBRARY_PATH on Unix). If multiple versions of onnxruntime are installed on the system this can make them find the wrong libraries and lead to undefined behavior.

#### Loading the shared providers

Shared provider libraries are loaded by the onnxruntime code (do not load or depend on them in your client code). The API for registering shared or non shared providers is identical, the difference is that shared ones will be loaded at runtime when the provider is added to the session options (through a call like OrtSessionOptionsAppendExecutionProvider_OpenVINO or SessionOptionsAppendExecutionProvider_OpenVINO in the C API).
If a shared provider library cannot be loaded (if the file doesn't exist, or its dependencies don't exist or not in the path) then an error will be returned.

The onnxruntime code will look for the provider shared libraries in the same location as the onnxruntime shared library is (or the executable statically linked to the static library version).

---

### Execution Providers

#### CUDA

##### Prerequisites

* Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn)
  * ONNX Runtime is built and tested with CUDA 10.2 and cuDNN 8.0.3 using Visual Studio 2019 version 16.7.
    ONNX Runtime can also be built with CUDA versions from 10.1 up to 11.0, and cuDNN versions from 7.6 up to 8.0.
  * The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home` parameter
  * The path to the cuDNN installation (include the `cuda` folder in the path) must be provided via the cuDNN_PATH environment variable, or `--cudnn_home` parameter. The cuDNN path should contain `bin`, `include` and `lib` directories.
  * The path to the cuDNN bin directory must be added to the PATH environment variable so that cudnn64_8.dll is found.


##### Build Instructions

###### Windows

```
.\build.bat --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path>
```

###### Linux

```
./build.sh --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path>
```

A Dockerfile is available [here](https://github.com/microsoft/onnxruntime/blob/master/dockerfiles#cuda).

##### Notes
* Depending on compatibility between the CUDA, cuDNN, and Visual Studio 2017 versions you are using, you may need to explicitly install an earlier version of the MSVC toolset.
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

#### TensorRT

See more information on the TensorRT Execution Provider [here](../reference/execution-providers/TensorRT-ExecutionProvider.md).

##### Prerequisites

* Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn)
   * The TensorRT execution provider for ONNX Runtime is built and tested with CUDA 11.0 and cuDNN 8.0.
   * The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home` parameter. The CUDA path should contain `bin`, `include` and `lib` directories.
   * The path to the CUDA `bin` directory must be added to the PATH environment variable so that `nvcc` is found.
   * The path to the cuDNN installation (path to folder that contains libcudnn.so) must be provided via the cuDNN_PATH environment variable, or `--cudnn_home` parameter.
 * Install [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download)
   * The TensorRT execution provider for ONNX Runtime is built on TensorRT 7.1 and is tested with TensorRT 7.1.3.4.
   * The path to TensorRT installation must be provided via the `--tensorrt_home` parameter.

##### Build Instructions
###### Windows
```
.\build.bat --cudnn_home <path to cuDNN home> --cuda_home <path to CUDA home> --use_tensorrt --tensorrt_home <path to TensorRT home>
```

###### Linux

```
./build.sh --cudnn_home <path to cuDNN e.g. /usr/lib/x86_64-linux-gnu/> --cuda_home <path to folder for CUDA e.g. /usr/local/cuda> --use_tensorrt --tensorrt_home <path to TensorRT home>
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#tensorrt)

---

##### NVIDIA Jetson TX1/TX2/Nano/Xavier

These instructions are for JetPack SDK 4.4.

1. Clone the ONNX Runtime repo on the Jetson host

    ```bash
    git clone --recursive https://github.com/microsoft/onnxruntime
    ```

2. Specify the CUDA compiler, or add its location to the PATH.

   Cmake can't automatically find the correct nvcc if it's not in the PATH.

    ```bash
    export CUDACXX="/usr/local/cuda/bin/nvcc"

    ```

    or:

    ```bash
    export PATH="/usr/local/cuda/bin:${PATH}"
    ```

3. Install the ONNX Runtime build dependencies on the Jetpack 4.4 host:

    ```bash
    sudo apt install -y --no-install-recommends \
      build-essential software-properties-common libopenblas-dev \
      libpython3.6-dev python3-pip python3-dev python3-setuptools python3-wheel
    ```

4. Cmake is needed to build ONNX Runtime. Because the minimum required version is 3.13,
   it is necessary to build CMake from source. Download Unix/Linux sources from https://cmake.org/download/
   and follow https://cmake.org/install/ to build from source. Version 3.17.5 and 3.18.4 have been tested on Jetson.

5. Build the ONNX Runtime Python wheel:

    ```bash
    ./build.sh --config Release --update --build --parallel --build_wheel \
    --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu
    ```

    Note: You may optionally build with experimental TensorRT support.

    ```bash
    ./build.sh --config Release --update --build --parallel --build_wheel \
    --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
    --tensorrt_home /usr/lib/aarch64-linux-gnu
    ```

---

#### DNNL and MKLML

See more information on DNNL and MKL-ML [here](../reference/execution-providers/DNNL-ExecutionProvider.md).

##### Build Instructions

The DNNL execution provider can be built for Intel CPU or GPU. To build for Intel GPU, install [Intel SDK for OpenCL Applications](https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html). Install the latest GPU driver - [Windows graphics driver](https://downloadcenter.intel.com/product/80939/Graphics), [Linux graphics compute runtime and OpenCL driver](https://github.com/intel/compute-runtime/releases).

###### Windows
`.\build.bat --use_dnnl`

###### Linux
`./build.sh --use_dnnl`

To build for Intel GPU, replace dnnl_opencl_root with the path of the Intel SDK for OpenCL Applications.

###### Windows

`.\build.bat --use_dnnl --dnnl_gpu_runtime ocl --dnnl_opencl_root "c:\program files (x86)\intelswtools\sw_dev_tools\opencl\sdk"`
###### Linux

`./build.sh --use_dnnl --dnnl_gpu_runtime ocl --dnnl_opencl_root "/opt/intel/sw_dev_tools/opencl-sdk"`s

---

#### OpenVINO

See more information on the OpenVINO Execution Provider [here](../reference/execution-providers/OpenVINO-ExecutionProvider.md).

##### Prerequisites

1. Install the Intel<sup>®</sup> Distribution of OpenVINO<sup>TM</sup> Toolkit **Release 2021.3** for the appropriate OS and target hardware:
   * [Linux - CPU, GPU, VPU, VAD-M](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux)
   * [Linux - FPGA](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux-fpga)
   * [Windows - CPU, GPU, VPU, VAD-M](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-windows).

   Follow [documentation](https://docs.openvinotoolkit.org/2021.3/index.html) for detailed instructions.

  *2021.3 is the recommended OpenVINO version. [OpenVINO 2020.3](https://docs.openvinotoolkit.org/2020.3/index.html) is minimal OpenVINO version requirement.*
  *The minimum ubuntu version to support 2021.3 is 18.04.*

2. Configure the target hardware with specific follow on instructions:
   * To configure Intel<sup>®</sup> Processor Graphics(GPU) please follow these instructions: [Windows](https://docs.openvinotoolkit.org/2021.3/openvino_docs_install_guides_installing_openvino_windows.html#Install-GPU), [Linux](https://docs.openvinotoolkit.org/2021.3/openvino_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps)
   * To configure Intel<sup>®</sup> Movidius<sup>TM</sup> USB, please follow this getting started guide: [Linux](https://docs.openvinotoolkit.org/2021.3/openvino_docs_install_guides_installing_openvino_linux.html#additional-NCS-steps)
   * To configure Intel<sup>®</sup> Vision Accelerator Design based on 8 Movidius<sup>TM</sup> MyriadX VPUs, please follow this configuration guide: [Windows](https://docs.openvinotoolkit.org/2021.3/openvino_docs_install_guides_installing_openvino_windows.html#hddl-myriad), [Linux](https://docs.openvinotoolkit.org/2021.3/openvino_docs_install_guides_installing_openvino_linux.html#install-VPU). Follow steps 3 and 4 to complete the configuration.
   * To configure Intel<sup>®</sup> Vision Accelerator Design with an Intel<sup>®</sup> Arria<sup>®</sup> 10 FPGA, please follow this configuration guide: [Linux](https://docs.openvinotoolkit.org/2021.3/openvino_docs_install_guides_installing_openvino_linux_fpga.html)

3. Initialize the OpenVINO environment by running the setupvars script as shown below:
   * For Linux run:
   ```
      $ source <openvino_install_directory>/bin/setupvars.sh
   ```
   * For Windows run:
   ```
      C:\ <openvino_install_directory>\bin\setupvars.bat
   ```

4. Extra configuration step for Intel<sup>®</sup> Vision Accelerator Design based on 8 Movidius<sup>TM</sup> MyriadX VPUs:
   * After setting the environment using setupvars script, follow these steps to change the default scheduler of VAD-M to Bypass:
      * Edit the hddl_service.config file from $HDDL_INSTALL_DIR/config/hddl_service.config and change the field "bypass_device_number" to 8.
      * Restart the hddl daemon for the changes to take effect.
      * Note that if OpenVINO was installed with root permissions, this file has to be changed with the same permissions.


##### Build Instructions
###### Windows

```
.\build.bat --config RelWithDebInfo --use_openvino <hardware_option> --build_shared_lib
```

*Note: The default Windows CMake Generator is Visual Studio 2017, but you can also use the newer Visual Studio 2019 by passing `--cmake_generator "Visual Studio 16 2019"` to `.\build.bat`*

###### Linux

```bash
./build.sh --config RelWithDebInfo --use_openvino <hardware_option> --build_shared_lib
```

* `--use_openvino` builds the OpenVINO Execution Provider in ONNX Runtime.
* `<hardware_option>`: Specifies the default hardware target for building OpenVINO Execution Provider. This can be overriden dynamically at runtime with another option (refer to [OpenVINO-ExecutionProvider.md](../reference/execution-providers/OpenVINO-ExecutionProvider.md) for more details on dynamic device selection). Below are the options for different Intel target devices.

| Hardware Option | Target Device |
| --------------- | ------------------------|
| <code>CPU_FP32</code> | Intel<sup>®</sup> CPUs |
| <code>GPU_FP32</code> | Intel<sup>®</sup> Integrated Graphics |
| <code>GPU_FP16</code> | Intel<sup>®</sup> Integrated Graphics with FP16 quantization of models |
| <code>MYRIAD_FP16</code> | Intel<sup>®</sup> Movidius<sup>TM</sup> USB sticks | 
| <code>VAD-M_FP16</code> | Intel<sup>®</sup> Vision Accelerator Design based on 8 Movidius<sup>TM</sup> MyriadX VPUs |
| <code>VAD-F_FP32</code> | Intel<sup>®</sup> Vision Accelerator Design with an Intel<sup>®</sup> Arria<sup>®</sup> 10 FPGA |
| <code>HETERO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>...</code> | All Intel<sup>®</sup> silicons mentioned above |
| <code>MULTI:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>...</code> | All Intel<sup>®</sup> silicons mentioned above |

Specifying Hardware Target for HETERO or Multi-Device Build:

HETERO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>...
The <DEVICE_TYPE> can be any of these devices from this list ['CPU','GPU','MYRIAD','FPGA','HDDL']

A minimum of two DEVICE_TYPE'S should be specified for a valid HETERO or Multi-Device Build.

Example:
HETERO:MYRIAD,CPU  HETERO:HDDL,GPU,CPU  MULTI:MYRIAD,GPU,CPU

For more information on OpenVINO Execution Provider&#39;s ONNX Layer support, Topology support, and Intel hardware enabled, please refer to the document [OpenVINO-ExecutionProvider.md](../reference/execution-providers/OpenVINO-ExecutionProvider.md) in <code>$onnxruntime_root/docs/execution_providers</code>

---

#### NUPHAR
See more information on the Nuphar Execution Provider [here](../reference/execution-providers/Nuphar-ExecutionProvider.md).

##### Prerequisites
* The Nuphar execution provider for ONNX Runtime is built and tested with LLVM 9.0.0. Because of TVM's requirement when building with LLVM, you need to build LLVM from source. To build the debug flavor of ONNX Runtime, you need the debug build of LLVM.
   * Windows (Visual Studio 2017):
   ```
   REM download llvm source code 9.0.0 and unzip to \llvm\source\path, then install to \llvm\install\path
   cd \llvm\source\path
   mkdir build
   cd build
   cmake .. -G "Visual Studio 15 2017 Win64" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_DIA_SDK=OFF
   msbuild llvm.sln /maxcpucount /p:Configuration=Release /p:Platform=x64
   cmake -DCMAKE_INSTALL_PREFIX=\llvm\install\path -DBUILD_TYPE=Release -P cmake_install.cmake
   ```

*Note that following LLVM cmake patch is necessary to make the build work on Windows, Linux does not need to apply the patch.*
The patch is to fix the linking warning LNK4199 caused by this [LLVM commit](https://github.com/llvm-mirror/llvm/commit/148f823e4845c9a13faea62e3105abb80b39e4bc)

```
diff --git "a/lib\\Support\\CMakeLists.txt" "b/lib\\Support\\CMakeLists.txt"
index 7dfa97c..6d99e71 100644
--- "a/lib\\Support\\CMakeLists.txt"
+++ "b/lib\\Support\\CMakeLists.txt"
@@ -38,12 +38,6 @@ elseif( CMAKE_HOST_UNIX )
   endif()
 endif( MSVC OR MINGW )

-# Delay load shell32.dll if possible to speed up process startup.
-set (delayload_flags)
-if (MSVC)
-  set (delayload_flags delayimp -delayload:shell32.dll -delayload:ole32.dll)
-endif()
-
 # Link Z3 if the user wants to build it.
 if(LLVM_WITH_Z3)
   set(Z3_LINK_FILES ${Z3_LIBRARIES})
@@ -187,7 +181,7 @@ add_llvm_library(LLVMSupport
   ${LLVM_MAIN_INCLUDE_DIR}/llvm/ADT
   ${LLVM_MAIN_INCLUDE_DIR}/llvm/Support
   ${Backtrace_INCLUDE_DIRS}
-  LINK_LIBS ${system_libs} ${delayload_flags} ${Z3_LINK_FILES}
+  LINK_LIBS ${system_libs} ${Z3_LINK_FILES}
   )

 set_property(TARGET LLVMSupport PROPERTY LLVM_SYSTEM_LIBS "${system_libs}")
```
   * Linux
   Download llvm source code 9.0.0 and unzip to /llvm/source/path, then install to /llvm/install/path
   ```
   cd /llvm/source/path
   mkdir build
   cd build
   cmake .. -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   cmake -DCMAKE_INSTALL_PREFIX=/llvm/install/path -DBUILD_TYPE=Release -P cmake_install.cmake
   ```

##### Build Instructions
###### Windows
```
.\build.bat --llvm_path=\llvm\install\path\lib\cmake\llvm --use_mklml --use_nuphar --build_shared_lib --build_csharp --enable_pybind --config=Release
```

* These instructions build the release flavor. The Debug build of LLVM would be needed to build with the Debug flavor of ONNX Runtime.

###### Linux:
```
./build.sh --llvm_path=/llvm/install/path/lib/cmake/llvm --use_mklml --use_nuphar --build_shared_lib --build_csharp --enable_pybind --config=Release
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#nuphar).


---

#### DirectML
See more information on the DirectML execution provider [here](../reference/execution-providers/DirectML-ExecutionProvider.md).
##### Windows
```
.\build.bat --use_dml
```
##### Notes
The DirectML execution provider supports building for both x64 and x86 architectures. DirectML is only supported on Windows.

---

#### ARM Compute Library
See more information on the ACL Execution Provider [here](../reference/execution-providers/ACL-ExecutionProvider.md).

##### Prerequisites
* Supported backend: i.MX8QM Armv8 CPUs
* Supported BSP: i.MX8QM BSP
  * Install i.MX8QM BSP: `source fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4*.sh`
* Set up the build environment
```
source /opt/fsl-imx-xwayland/4.*/environment-setup-aarch64-poky-linux
alias cmake="/usr/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
```
* See [Build ARM](#ARM) below for information on building for ARM devices

##### Build Instructions

1. Configure ONNX Runtime with ACL support:
```
cmake ../onnxruntime-arm-upstream/cmake -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc -Donnxruntime_RUN_ONNX_TESTS=OFF -Donnxruntime_GENERATE_TEST_REPORTS=ON -Donnxruntime_DEV_MODE=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Donnxruntime_USE_CUDA=OFF -Donnxruntime_USE_NSYNC=OFF -Donnxruntime_CUDNN_HOME= -Donnxruntime_USE_JEMALLOC=OFF -Donnxruntime_ENABLE_PYTHON=OFF -Donnxruntime_BUILD_CSHARP=OFF -Donnxruntime_BUILD_SHARED_LIB=ON -Donnxruntime_USE_EIGEN_FOR_BLAS=ON -Donnxruntime_USE_OPENBLAS=OFF -Donnxruntime_USE_ACL=ON -Donnxruntime_USE_DNNL=OFF -Donnxruntime_USE_MKLML=OFF -Donnxruntime_USE_OPENMP=ON -Donnxruntime_USE_TVM=OFF -Donnxruntime_USE_LLVM=OFF -Donnxruntime_ENABLE_MICROSOFT_INTERNAL=OFF -Donnxruntime_USE_BRAINSLICE=OFF -Donnxruntime_USE_NUPHAR=OFF -Donnxruntime_USE_EIGEN_THREADPOOL=OFF -Donnxruntime_BUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
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

##### Native Build Instructions (validated on Jetson Nano and Jetson Xavier)

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

#### ArmNN

See more information on the ArmNN Execution Provider [here](../reference/execution-providers/ArmNN-ExecutionProvider.md).

##### Prerequisites

* Supported backend: i.MX8QM Armv8 CPUs
* Supported BSP: i.MX8QM BSP
  * Install i.MX8QM BSP: `source fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4*.sh`
* Set up the build environment

```bash
source /opt/fsl-imx-xwayland/4.*/environment-setup-aarch64-poky-linux
alias cmake="/usr/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
```

* See [Build ARM](#ARM) below for information on building for ARM devices

##### Build Instructions

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

#### RKNPU
See more information on the RKNPU Execution Provider [here](../reference/execution-providers/RKNPU-ExecutionProvider.md).

##### Prerequisites

* Supported platform: RK1808 Linux
* See [Build ARM](#ARM) below for information on building for ARM devices
* Use gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu instead of gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf, and modify CMAKE_CXX_COMPILER & CMAKE_C_COMPILER in tool.cmake:

```
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
```

##### Build Instructions
###### Linux

1. Download [rknpu_ddk](#https://github.com/airockchip/rknpu_ddk.git) to any directory.

1. Build ONNX Runtime library and test:

    ```bash
    ./build.sh --arm --use_rknpu --parallel --build_shared_lib --build_dir build_arm --config MinSizeRel --cmake_extra_defines RKNPU_DDK_PATH=<Path To rknpu_ddk> CMAKE_TOOLCHAIN_FILE=<Path To tool.cmake> ONNX_CUSTOM_PROTOC_EXECUTABLE=<Path To protoc>
    ```

1. Deploy ONNX runtime and librknpu_ddk.so on the RK1808 board:

    ```bash
    libonnxruntime.so.1.2.0
    onnxruntime_test_all
    rknpu_ddk/lib64/librknpu_ddk.so
    ```

---

#### Vitis-AI
See more information on the Xilinx Vitis-AI execution provider [here](../reference/execution-providers/Vitis-AI-ExecutionProvider.md).

For instructions to setup the hardware environment: [Hardware setup](../reference/execution-providers/Vitis-AI-ExecutionProvider.md#Hardware-setup)

##### Linux

```bash
./build.sh --use_vitisai
```

##### Notes
The Vitis-AI execution provider is only supported on Linux.

### Options
#### OpenMP
##### Build Instructions
###### Windows

```powershell
.\build.bat --use_openmp
```

###### Linux/macOS

```bash
./build.sh --use_openmp
```

---

#### OpenBLAS
##### Prerequisites

* OpenBLAS
   * Windows: See build instructions [here](https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio#build-openblas-for-universal-windows-platform)
   * Linux: Install the libopenblas-dev package `sudo apt-get install libopenblas-dev`

##### Build Instructions
###### Windows

```
.\build.bat --use_openblas
```

###### Linux

```bash
./build.sh --use_openblas
```

---

#### DebugNodeInputsOutputs
OnnxRuntime supports build options for enabling debugging of intermediate tensor shapes and data.

##### Build Instructions
Set onnxruntime_DEBUG_NODE_INPUTS_OUTPUT to build with this enabled.

###### Linux

```bash
./build.sh --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
```

###### Windows

```
.\build.bat --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
```

##### Configuration

The debug dump behavior can be controlled with several environment variables.
See [onnxruntime/core/framework/debug_node_inputs_outputs_utils.h](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/framework/debug_node_inputs_outputs_utils.h) for details.

###### Examples

To specify that node output data should be dumped (to stdout by default), set this environment variable:

```
ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1
```

To specify that node output data should be dumped to files for nodes with name "Foo" or "Bar", set these environment variables:

```
ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1
ORT_DEBUG_NODE_IO_NAME_FILTER="Foo;Bar"
ORT_DEBUG_NODE_IO_DUMP_DATA_TO_FILES=1
```

---

### Architectures
#### 64-bit x86

Also known as [x86_64](https://en.wikipedia.org/wiki/X86-64) or AMD64. This is the default.

#### 32-bit x86
##### Build Instructions
###### Windows
* add `--x86` argument when launching `.\build.bat`

###### Linux
(Not officially supported)

---

#### ARM

There are a few options for building for ARM.

* [Cross compiling for ARM with simulation (Linux/Windows)](#Cross-compiling-for-ARM-with-simulation-LinuxWindows) - **Recommended**;  Easy, slow
* [Cross compiling on Linux](#Cross-compiling-on-Linux) - Difficult, fast
* [Native compiling on Linux ARM device](#Native-compiling-on-Linux-ARM-device) - Easy, slower
* [Cross compiling on Windows](#Cross-compiling-on-Windows)

##### Cross compiling for ARM with simulation (Linux/Windows)

*EASY, SLOW, RECOMMENDED*

This method rely on qemu user mode emulation. It allows you to compile using a desktop or cloud VM through instruction level simulation. You'll run the build on x86 CPU and translate every ARM instruction to x86. This is much faster than compiling natively on a low-end ARM device and avoids out-of-memory issues that may be encountered. The resulting ONNX Runtime Python wheel (.whl) file is then deployed to an ARM device where it can be invoked in Python 3 scripts.

Here is [an example for Raspberrypi3 and Raspbian](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles/README.md#arm-32v7). Note: this does not work for Raspberrypi 1 or Zero, and if your operating system is different from what the dockerfile uses, it also may not work.

The build process can take hours.

##### Cross compiling on Linux

*Difficult, fast*

This option is very fast and allows the package to be built in minutes, but is challenging to setup. If you have a large code base (e.g. you are adding a new execution provider to onnxruntime), this may be the only feasible method.

1. Get the corresponding toolchain.

    TLDR; Go to https://www.linaro.org/downloads/, get "64-bit Armv8 Cortex-A, little-endian" and "Linux Targeted", not "Bare-Metal Targeted". Extract it to your build machine and add the bin folder to your $PATH env. Then skip this part.

    You can use [GCC](https://gcc.gnu.org/) or [Clang](http://clang.llvm.org/). Both work, but instructions here are based on GCC.

    In GCC terms:
    * "build" describes the type of system on which GCC is being configured and compiled
    * "host" describes the type of system on which GCC runs.
    * "target" to describe the type of system for which GCC produce code

    When not cross compiling, usually "build" = "host" = "target". When you do cross compile, usually "build" = "host" != "target". For example, you may build GCC on x86_64, then run GCC on x86_64, then generate binaries that target aarch64. In this case,"build" = "host" = x86_64 Linux, target is aarch64 Linux.

    You can either build GCC from source code by yourself, or get a prebuilt one from a vendor like Ubuntu, linaro. Choosing the same compiler version as your target operating system is best. If ths is not possible, choose the latest stable one and statically link to the GCC libs.

    When you get the compiler, run `aarch64-linux-gnu-gcc -v` This should produce an output like below:

    ```bash
    Using built-in specs.
    COLLECT_GCC=/usr/bin/aarch64-linux-gnu-gcc
    COLLECT_LTO_WRAPPER=/usr/libexec/gcc/aarch64-linux-gnu/9/lto-wrapper
    Target: aarch64-linux-gnu
    Configured with: ../gcc-9.2.1-20190827/configure --bindir=/usr/bin --build=x86_64-redhat-linux-gnu --datadir=/usr/share --disable-decimal-float --disable-dependency-tracking --disable-gold --disable-libgcj --disable-libgomp --disable-libmpx --disable-libquadmath --disable-libssp --disable-libunwind-exceptions --disable-shared --disable-silent-rules --disable-sjlj-exceptions --disable-threads --with-ld=/usr/bin/aarch64-linux-gnu-ld --enable-__cxa_atexit --enable-checking=release --enable-gnu-unique-object --enable-initfini-array --enable-languages=c,c++ --enable-linker-build-id --enable-lto --enable-nls --enable-obsolete --enable-plugin --enable-targets=all --exec-prefix=/usr --host=x86_64-redhat-linux-gnu --includedir=/usr/include --infodir=/usr/share/info --libexecdir=/usr/libexec --localstatedir=/var --mandir=/usr/share/man --prefix=/usr --program-prefix=aarch64-linux-gnu- --sbindir=/usr/sbin --sharedstatedir=/var/lib --sysconfdir=/etc --target=aarch64-linux-gnu --with-bugurl=http://bugzilla.redhat.com/bugzilla/ --with-gcc-major-version-only --with-isl --with-newlib --with-plugin-ld=/usr/bin/aarch64-linux-gnu-ld --with-sysroot=/usr/aarch64-linux-gnu/sys-root --with-system-libunwind --with-system-zlib --without-headers --enable-gnu-indirect-function --with-linker-hash-style=gnu
    Thread model: single
    gcc version 9.2.1 20190827 (Red Hat Cross 9.2.1-3) (GCC)
    ```

    Check the value of `--build`, `--host`, `--target`, and if it has special args like `--with-arch=armv8-a`, `--with-arch=armv6`, `--with-tune=arm1176jz-s`, `--with-fpu=vfp`, `--with-float=hard`.

    You must also know what kind of flags your target hardware need, which can differ greatly. For example, if you just get the normal ARMv7 compiler and use it for Raspberry Pi V1 directly, it won't work because Raspberry Pi only has ARMv6. Generally every hardware vendor will provide a toolchain; check how that one was built.

    A target env is identifed by:

    * Arch: x86_32, x86_64, armv6,armv7,arvm7l,aarch64,...
    * OS: bare-metal or linux.
    * Libc: gnu libc/ulibc/musl/...
    * ABI: ARM has mutilple ABIs like eabi, eabihf...

    You can get all these information from the previous output, please be sure they are all correct.

2. Get a pre-compiled protoc:

   Get this from https://github.com/protocolbuffers/protobuf/releases/download/v3.11.2/protoc-3.11.2-linux-x86_64.zip and unzip after downloading.
   The version must match the one onnxruntime is using. Currently we are using 3.11.2.

3. (Optional) Setup sysroot to enable python extension. *Skip if not using Python.*

    Dump the root file system of the target operating system to your build machine. We'll call that folder "sysroot" and use it for build onnxruntime python extension. Before doing that, you should install python3 dev package(which contains the C header files) and numpy python package on the target machine first.

    Below are some examples.

    If the target OS is raspbian-buster, please download the RAW image from [their website](https://www.raspberrypi.org/downloads/raspbian/) then run:

    ```bash
    $ fdisk -l 2020-02-13-raspbian-buster.img
    ```

    Disk 2020-02-13-raspbian-buster.img: 3.54 GiB, 3787456512 bytes, 7397376 sectors
    Units: sectors of 1 * 512 = 512 bytes
    Sector size (logical/physical): 512 bytes / 512 bytes
    I/O size (minimum/optimal): 512 bytes / 512 bytes
    Disklabel type: dos
    Disk identifier: 0xea7d04d6

    | Device                          | Boot | Start  | End     | Sectors | Size | Id | Type            |
    |---------------------------------|------|--------|---------|---------|------|----|-----------------|
    | 2020-02-13-raspbian-buster.img1 |      | 8192   | 532479  | 524288  | 256M | c  | W95 FAT32 (LBA) |
    | 2020-02-13-raspbian-buster.img2 |      | 532480 | 7397375 | 6864896 | 3.3G | 83 | Linux           |

    You'll find the the root partition starts at the 532480 sector, which is 532480 \* 512=272629760 bytes from the beginning.

    Then run:

    ```bash
    $ mkdir /mnt/pi
    $ mount -r -o loop,offset=272629760 2020-02-13-raspbian-buster.img /mnt/pi
    ```

    You'll see all raspbian files at /mnt/pi. However you can't use it yet. Because some of the symlinks are broken, you must fix them first.

    In /mnt/pi, run

    ```bash
    $ find . -type l -exec realpath  {} \; |grep 'No such file'
    ```

    It will show which are broken.
    Then you can fix them by running:

    ```bash
    $ mkdir /mnt/pi2
    $ cd /mnt/pi2
    $ sudo tar -C /mnt/pi -cf - . | sudo tar --transform 'flags=s;s,^/,/mnt/pi2/,' -xf -
    ```

    Then /mnt/pi2 is the sysroot folder you'll use in the next step.

    If the target OS is Ubuntu, you can get an image from [https://cloud-images.ubuntu.com/](https://cloud-images.ubuntu.com/). But that image is in qcow2 format. Please convert it before run fdisk and mount.

    ```bash
    qemu-img convert -p -O raw ubuntu-18.04-server-cloudimg-arm64.img ubuntu.raw
    ```

    The remaining part is similar to raspbian.

    If the target OS is manylinux2014, you can get it by:
    Install qemu-user-static from apt or dnf.
    Then run the docker

    Ubuntu:

    ```bash
    docker run -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -it --rm quay.io/pypa/manylinux2014_aarch64 /bin/bash
    ```

    The "-v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static" arg is not needed on Fedora.

    Then, inside the docker, run

    ```bash
    cd /opt/python
    ./cp35-cp35m/bin/python -m pip install numpy==1.16.6
    ./cp36-cp36m/bin/python -m pip install numpy==1.16.6
    ./cp37-cp37m/bin/python -m pip install numpy==1.16.6
    ./cp38-cp38/bin/python -m pip install numpy==1.16.6
    ```

    These commands will take a few hours because numpy doesn't have a prebuilt package yet. When completed, open a second window and run

    ```bash
    docker ps
    ```

    From the output:

    ```
    CONTAINER ID        IMAGE                                COMMAND             CREATED             STATUS              PORTS               NAMES
    5a796e98db05        quay.io/pypa/manylinux2014_aarch64   "/bin/bash"         3 minutes ago       Up 3 minutes                            affectionate_cannon
    ```

    You'll see the docker instance id is: 5a796e98db05. Use the following command to export the root filesystem as the sysroot for future use.

    ```bash
    docker export 5a796e98db05 -o manylinux2014_aarch64.tar
    ```

4. Generate CMake toolchain file
    Save the following content as tool.cmake

    ```cmake
    SET(CMAKE_SYSTEM_NAME Linux)
    SET(CMAKE_SYSTEM_VERSION 1)
    SET(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    SET(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
    SET(CMAKE_FIND_ROOT_PATH /mnt/pi)
    ```

    If you don't have a sysroot, you can delete the last line.

5.  Run CMake and make

    Append `-DONNX_CUSTOM_PROTOC_EXECUTABLE=/path/to/protoc -DCMAKE_TOOLCHAIN_FILE=path/to/tool.cmake` to your cmake args, run cmake and make to build it. If you want to build Python package as well, you can use cmake args like:

    ```bash
    -Donnxruntime_GCC_STATIC_CPP_RUNTIME=ON -DCMAKE_BUILD_TYPE=Release -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_TOOLCHAIN_FILE=path/to/tool.cmake -Donnxruntime_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/mnt/pi/usr/bin/python3 -Donnxruntime_BUILD_SHARED_LIB=OFF -Donnxruntime_DEV_MODE=OFF -DONNX_CUSTOM_PROTOC_EXECUTABLE=/path/to/protoc "-DPYTHON_INCLUDE_DIR=/mnt/pi/usr/include;/mnt/pi/usr/include/python3.7m" -DNUMPY_INCLUDE_DIR=/mnt/pi/folder/to/numpy/headers
    ```

    After running cmake, run

    ```bash
    $ make
    ```

6.  (Optional) Build Python package

    Copy the setup.py file from the source folder to the build folder and run

    ```bash
    python3 setup.py bdist_wheel -p linux_aarch64
    ```

    If targeting manylinux, unfortunately their tools do not work in the cross-compiling scenario. Run it in a docker like:

    ```bash
    docker run  -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -v `pwd`:/tmp/a -w /tmp/a --rm quay.io/pypa/manylinux2014_aarch64 /opt/python/cp37-cp37m/bin/python3 setup.py bdist_wheel
    ```

    This is not needed if you only want to target a specfic Linux distribution (i.e. Ubuntu).

##### Native compiling on Linux ARM device

*Easy, slower*

Docker build runs on a Raspberry Pi 3B with Raspbian Stretch Lite OS (Desktop version will run out memory when linking the .so file) will take 8-9 hours in total.

```bash
sudo apt-get update
sudo apt-get install -y \
    sudo \
    build-essential \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    git \
    tar

pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install --upgrade wheel
pip3 install numpy

# Build the latest cmake
mkdir /code
cd /code
wget https://cmake.org/files/v3.13/cmake-3.16.1.tar.gz;
tar zxf cmake-3.16.1.tar.gz

cd /code/cmake-3.16.1
./configure --system-curl
make
sudo make install

# Prepare onnxruntime Repo
cd /code
git clone --recursive https://github.com/Microsoft/onnxruntime

# Start the basic build
cd /code/onnxruntime
./build.sh --config MinSizeRel --update --build

# Build Shared Library
./build.sh --config MinSizeRel --build_shared_lib

# Build Python Bindings and Wheel
./build.sh --config MinSizeRel --enable_pybind --build_wheel

# Build Output
ls -l /code/onnxruntime/build/Linux/MinSizeRel/*.so
ls -l /code/onnxruntime/build/Linux/MinSizeRel/dist/*.whl
```

##### Cross compiling on Windows

**Using Visual C++ compilers**

1. Download and install Visual C++ compilers and libraries for ARM(64).
   If you have Visual Studio installed, please use the Visual Studio Installer (look under the section `Individual components` after choosing to `modify` Visual Studio) to download and install the corresponding ARM(64) compilers and libraries.

2. Use `.\build.bat` and specify `--arm` or `--arm64` as the build option to start building. Preferably use `Developer Command Prompt for VS` or make sure all the installed cross-compilers are findable from the command prompt being used to build using the PATH environmant variable.

---

#### Android

##### Prerequisites

The SDK and NDK packages can be installed via Android Studio or the sdkmanager command line tool.
Android Studio is more convenient but a larger installation.
The command line tools are smaller and usage can be scripted, but are  a little more complicated to setup. They also require a Java runtime environment to be available.

Resources:
- API levels: https://developer.android.com/guide/topics/manifest/uses-sdk-element.html
- Android ABIs: https://developer.android.com/ndk/guides/abis
- System Images: https://developer.android.com/topic/generic-system-image

###### Android Studio

1. Install Android Studio from https://developer.android.com/studio

1. Install any additional SDK Platforms if necessary

* File->Settings->Appearance & Behavior->System Settings->Android SDK to see what is currently installed
* Note that the SDK path you need to use as --android_sdk_path when building ORT is also on this configuration page
* Most likely you don't require additional SDK Platform packages as the latest platform can target earlier API levels.

1. Install an NDK version

* File->Settings->Appearance & Behavior->System Settings->Android SDK
* 'SDK Tools' tab
  * Select 'Show package details' checkbox at the bottom to see specific versions. By default the latest will be installed which should be fine.
* The NDK path will be the 'ndk/{version}' subdirectory of the SDK path shown
  * e.g. if 21.1.6352462 is installed it will be {SDK path}/ndk/21.1.6352462

###### sdkmanager from command line tools

* If necessary install the Java Runtime Environment and set the JAVA_HOME environment variable to point to it
  * https://www.java.com/en/download/
  * Windows note: You MUST install the 64-bit version (https://www.java.com/en/download/manual.jsp) otherwise sdkmanager will only list x86 packages
      and the latest NDK is x64 only.
* For sdkmanager to work it needs a certain directory structure. First create the top level directory for the Android infrastructure.
  * in our example we'll call that `.../Android/`
* Download the command line tools from the 'Command line tools only' section towards the bottom of https://developer.android.com/studio
* Create a directory called 'cmdline-tools' under your top level directory
  * giving `.../Android/cmdline-tools`
* Extract the 'tools' directory from the command line tools zip file into this directory
  * giving `.../Android/cmdline-tools/tools`
  * Windows note: preferably extract using 7-zip. If using the built in Windows zip extract tool you will need to fix the directory structure by moving the jar files from `tools\lib\_` up to `tools\lib`
    * See https://stackoverflow.com/questions/27364963/could-not-find-or-load-main-class-com-android-sdkmanager-main
* You should now be able to run Android/cmdline-tools/bin/sdkmanager[.bat] successfully
  * if you see an error about it being unable to save settings and the sdkmanager help text,
      your directory structure is incorrect.
  * see the final steps in this answer to double check: https://stackoverflow.com/a/61176718

* Run `.../Android/cmdline-tools/bin/sdkmanager --list` to see the packages available

* Install the SDK Platform
  * Generally installing the latest is fine. You pick an API level when compiling the code and the latest platform will support many recent API levels e.g.

    ```
    sdkmanager --install "platforms;android-29"
    ```

  * This will install into the 'platforms' directory of our top level directory, the `Android` directory in our example
  * The SDK path to use as `--android_sdk_path` when building is this top level directory

* Install the NDK
  * Find the available NDK versions by running `sdkmanager --list`
  * Install
    * you can install a specific version or the latest (called 'ndk-bundle') e.g. `sdkmanager --install "ndk;21.1.6352462"`
    * NDK path in our example with this install would be `.../Android/ndk/21.1.6352462`
    * NOTE: If you install the ndk-bundle package the path will be `.../Android/ndk-bundle` as there's no version number

##### Android Build Instructions

###### Cross compiling on Windows

The [Ninja](https://ninja-build.org/) generator needs to be used to build on Windows as the Visual Studio generator doesn't support Android.

```powershell
./build.bat --android --android_sdk_path <android sdk path> --android_ndk_path <android ndk path> --android_abi <android abi, e.g., arm64-v8a (default) or armeabi-v7a> --android_api <android api level, e.g., 27 (default)> --cmake_generator Ninja
```

e.g. using the paths from our example

```
./build.bat --android --android_sdk_path .../Android --android_ndk_path .../Android/ndk/21.1.6352462 --android_abi arm64-v8a --android_api 27 --cmake_generator Ninja
```

###### Cross compiling on Linux and macOS

```bash
./build.sh --android --android_sdk_path <android sdk path> --android_ndk_path <android ndk path> --android_abi <android abi, e.g., arm64-v8a (default) or armeabi-v7a> --android_api <android api level, e.g., 27 (default)>
```

###### Build Android Archive (AAR)

Android Archive (AAR) files, which can be imported directly in Android Studio, will be generated in `<your_build_dir>/java/build/android/outputs/aar`, by using the above building commands with `--build_java`

To build on Windows with `--build_java` enabled you must also:

* set JAVA_HOME to the path to your JDK install
  * this could be the JDK from Android Studio, or a [standalone JDK install](https://www.oracle.com/java/technologies/javase-downloads.html)
  * e.g. Powershell: `$env:JAVA_HOME="C:\Program Files\Java\jdk-15"` CMD: `set JAVA_HOME=C:\Program Files\Java\jdk-15`
* install [Gradle](https://gradle.org/install/) and add the directory to the PATH
  * e.g. Powershell: `$env:PATH="$env:PATH;C:\Gradle\gradle-6.6.1\bin"` CMD: `set PATH=%PATH%;C:\Gradle\gradle-6.6.1\bin`
* run the build from an admin window
  * the Java build needs permissions to create a symlink, which requires an admin window

##### Android NNAPI Execution Provider

If you want to use NNAPI Execution Provider on Android, see [NNAPI Execution Provider](../reference/execution-providers/NNAPI-ExecutionProvider.md).

###### Build Instructions

Android NNAPI Execution Provider can be built using building commands in [Android Build instructions](#android-build-instructions) with `--use_nnapi`

---

#### iOS

##### Prerequisites

* A Mac computer with latest macOS
* Xcode, https://developer.apple.com/xcode/
* CMake, https://cmake.org/download/
* Python 3, https://www.python.org/downloads/mac-osx/

##### General Info

* iOS Platforms

  The following two platforms are supported
  * iOS device (iPhone, iPad) with arm64 architecture
  * iOS simulator with x86_64 architecture

  The following platforms are *not* supported
  * armv7
  * armv7s
  * i386 architectures
  * tvOS
  * watchOS platforms are not currently supported.

* apple_deploy_target

  Specify the minimum version of the target platform (iOS) on which the target binaries are to be deployed.

* Code Signing

  If the development team ID and/or the code sign identity which has a valid code signing certificate is specified, Xcode will code sign the onnxruntime library in the building process, otherwise, the onnxruntime will be built without code signing. It may be required or desired to code sign the library for iOS devices. For more information, see [Code Signing](https://developer.apple.com/support/code-signing/).

##### iOS Build Instructions

Run one of the following build scripts from the ONNX Runtime repository root,
###### Cross build for iOS simulator

```bash
./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --ios_sysroot iphonesimulator --osx_arch x86_64 --apple_deploy_target <minimal iOS version>
```

###### Cross build for iOS device

```bash
./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --ios_sysroot iphoneos --osx_arch arm64 --apple_deploy_target <minimal iOS version>
```

###### Cross build for iOS device and code sign the library using development team ID

```bash
./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --ios_sysroot iphoneos --osx_arch arm64 --apple_deploy_target <minimal iOS version> \
           --xcode_code_signing_team_id <Your Apple developmemt team ID>
```

###### Cross build for iOS device and code sign the library using code sign identity

```bash
./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --ios_sysroot iphoneos --osx_arch arm64 --apple_deploy_target <minimal iOS version> \
           --xcode_code_signing_identity <Your preferred code sign identity>
```

##### Build c/c++ dynamic framework

c/c++ dynamic framework can be built using the above building commands with `--build_apple_framework`, for details about Apple Framework, please see [Apple Framework Document](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPFrameworks/Concepts/WhatAreFrameworks.html#//apple_ref/doc/uid/20002303-BBCEIJFI)


##### iOS CoreML Execution Provider

If you want to use CoreML Execution Provider on iOS, see [CoreML Execution Provider](../reference/execution-providers/CoreML-ExecutionProvider.md).

###### Build Instructions

CoreML Execution Provider can be built using building commands in [iOS Build instructions](#ios-build-instructions) with `--use_coreml`

---

#### AMD MIGraphX

See more information on the MIGraphX Execution Provider [here](../reference/execution-providers/MIGraphX-ExecutionProvider.md).

##### Prerequisites

* Install [ROCM](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)
  * The MIGraphX execution provider for ONNX Runtime is built and tested with ROCM3.3
* Install [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX)
  * The path to MIGraphX installation must be provided via the `--migraphx_home parameter`.

##### Build Instructions

###### Linux

```bash
./build.sh --config <Release|Debug|RelWithDebInfo> --use_migraphx --migraphx_home <path to MIGraphX home>
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/blob/master/dockerfiles#migraphx)

---

## Training
### CPU

#### Build Instructions

To build ORT with training support add `--enable_training` build instruction.

All other build options are the same for inferencing as they are for training.

##### Windows

```
.\build.bat --config RelWithDebInfo --build_shared_lib --parallel --enable_training
```

The default Windows CMake Generator is Visual Studio 2017, but you can also use the newer Visual Studio 2019 by passing
`--cmake_generator "Visual Studio 16 2019"` to `.\build.bat`


##### Linux/macOS

```
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --enable_training
```

### GPU / CUDA
#### Prerequisites

The default NVIDIA GPU build requires CUDA runtime libraries installed on the system:

* [CUDA](https://developer.nvidia.com/cuda-toolkit) 10.2
* [cuDNN](https://developer.nvidia.com/cudnn) 8.0
* [NCCL](https://developer.nvidia.com/nccl) 2.7
* [OpenMPI](https://www.open-mpi.org/) 4.0.4
  * See [install_openmpi.sh](https://github.com/microsoft/onnxruntime/blob/master/tools/ci_build/github/linux/docker/scripts/install_openmpi.sh)

These dependency versions should reflect what is in [Dockerfile.training](https://github.com/microsoft/onnxruntime/blob/master/dockerfiles/Dockerfile.training).

#### Build instructions

1. Checkout this code repo with `git clone https://github.com/microsoft/onnxruntime`

2. Set the environment variables: *adjust the path for location your build machine*
    ```
    export CUDA_HOME=<location for CUDA libs> # e.g. /usr/local/cuda
    export CUDNN_HOME=<location for cuDNN libs> # e.g. /usr/local/cuda
    export CUDACXX=<location for NVCC> #e.g. /usr/local/cuda/bin/nvcc
    export PATH=<location for openmpi/bin/>:$PATH
    export LD_LIBRARY_PATH=<location for openmpi/lib/>:$LD_LIBRARY_PATH
    export MPI_CXX_INCLUDE_PATH=<location for openmpi/include/>
    source <location of the mpivars script> # e.g. /data/intel/impi/2018.3.222/intel64/bin/mpivars.sh
    ```

3. Create the ONNX Runtime wheel

   * Change to the ONNX Runtime repo base folder: `cd onnxruntime`
   * Run `./build.sh --enable_training --use_cuda --config=RelWithDebInfo --build_wheel`

    This produces the .whl file in `./build/Linux/RelWithDebInfo/dist` for ONNX Runtime Training.

### GPU / ROCM
#### Prerequisites

The default AMD GPU build requires ROCM software toolkit installed on the system:

* [ROCM](https://rocmdocs.amd.com/en/latest/)
* [OpenMPI](https://www.open-mpi.org/) 4.0.4
  * See [install_openmpi.sh](./tools/ci_build/github/linux/docker/scripts/install_openmpi.sh)

These dependency versions should reflect what is in [Dockerfile.training](./dockerfiles/Dockerfile.training).

#### Build instructions

1. Checkout this code repo with `git clone https://github.com/microsoft/onnxruntime`

2. Create the ONNX Runtime wheel

   * Change to the ONNX Runtime repo base folder: `cd onnxruntime`
   * Run `./build.sh --config RelWithDebInfo --enable_training --build_wheel --use_rocm --rocm_home /opt/rocm --nccl_home /opt/rocm --mpi_home <location for openmpi>`

    This produces the .whl file in `./build/Linux/RelWithDebInfo/dist` for ONNX Runtime Training.

### DNNL and MKLML

#### Build Instructions
##### Linux

`./build.sh --enable_training --use_dnnl`

##### Windows

`.\build.bat --enable_training --use_dnnl`

Add `--build_wheel` to build the ONNX Runtime wheel

This will produce a .whl file in `build/Linux/RelWithDebInfo/dist` for ONNX Runtime Training
