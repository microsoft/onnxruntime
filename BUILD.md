# Building ONNX Runtime
*Dockerfiles are available [here](./dockerfiles) to help you get started.*

*Pre-built packages are available at the locations indicated [here](https://github.com/microsoft/onnxruntime#official-builds).*

## Getting Started: Build the baseline CPU version of ONNX Runtime from source

### Pre-Requisites
* Checkout the source tree:
   ```
   git clone --recursive https://github.com/Microsoft/onnxruntime
   cd onnxruntime
   ```
* Install cmake-3.13 or higher from https://cmake.org/download/.


### Build Instructions
#### Windows
Open Developer Command Prompt for Visual Studio version you are going to use. This will properly setup the environment including paths to your compiler, linker, utilities and header files.
```
.\build.bat --config RelWithDebInfo --build_shared_lib --parallel
```
The default Windows CMake Generator is Visual Studio 2017, but you can also use the newer Visual Studio 2019 by passing `--cmake_generator "Visual Studio 16 2019"` to `.\build.bat`


#### Linux
```
./build.sh --config RelWithDebInfo --build_shared_lib --parallel
```

#### Notes

* Please note that these instructions build the debug build, which may have performance tradeoffs
* To build the version from each release (which include Windows, Linux, and Mac variants), see these .yml files for reference: [CPU](./tools/ci_build/github/azure-pipelines/nuget/cpu-esrp-pipeline.yml), [GPU](./tools/ci_build/github/azure-pipelines/nuget/gpu-esrp-pipeline.yml)
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

# Supported architectures and build environments

## Architectures

|           | x86_32       | x86_64       | ARM32v7      | ARM64        |
|-----------|:------------:|:------------:|:------------:|:------------:|
|Windows    | YES          | YES          |  YES         | YES          |
|Linux      | YES          | YES          |  YES         | YES          |
|Mac OS X   | NO           | YES          |  NO          | NO           |

## Environments

| OS          | Supports CPU | Supports GPU| Notes                              |
|-------------|:------------:|:------------:|------------------------------------|
|Windows 10   | YES          | YES         | VS2019 through the latest VS2015 are supported |
|Windows 10 <br/> Subsystem for Linux | YES         | NO        |         |
|Ubuntu 16.x  | YES          | YES         | Also supported on ARM32v7 (experimental) |

* GCC 4.x and below are not supported.

### OS/Compiler Matrix:

| OS/Compiler | Supports VC  | Supports GCC     |
|-------------|:------------:|:----------------:|
|Windows 10   | YES          | Not tested       |
|Linux        | NO           | YES(gcc>=4.8)    |

## System Requirements
For other system requirements and other dependencies, please see [this section](./README.md#system-requirements-pre-requisite-dependencies).

---
# Common Build Instructions
|Description|Command|Additional description|
|-----------|-----------|-----------|
|**Basic build**|build.bat (Windows)<br>./build.sh (Linux)||
|**Debug build**|--config RelWithDebInfo|Debug build|
|**Use OpenMP**|--use_openmp|OpenMP will parallelize some of the code for potential performance improvements. This is not recommended for running on single threads.|
|**Build using parallel processing**|--parallel|This is strongly recommended to speed up the build.|
|**Build Shared Library**|--build_shared_lib||
|**Build Python wheel**|--build_wheel||
|**Build C# and C packages**|--build_csharp||
|**Build WindowsML**|--use_winml<br>--use_dml<br>--build_shared_lib|WindowsML depends on DirectML and the OnnxRuntime shared library.|
|**Build Java package**|--build_java|Creates an onnxruntime4j.jar in the build directory, implies `--build_shared_lib`|


# Additional Build Instructions
The complete list of build options can be found by running `./build.sh (or .\build.bat) --help`

* [ONNX Runtime Server (Linux)](#Build-ONNX-Runtime-Server-on-Linux)

**Execution Providers**
* [NVIDIA CUDA](#CUDA)
* [NVIDIA TensorRT](#TensorRT)
* [Intel DNNL/MKL-ML](#DNNL-and-MKLML)
* [Intel nGraph](#nGraph)
* [Intel OpenVINO](#openvino)
* [Android NNAPI](#Android-NNAPI)
* [Nuphar Model Compiler](#Nuphar)
* [DirectML](#DirectML)
* [ARM Compute Library](#ARM-Compute-Library)

**Options**
* [OpenMP](#OpenMP)
* [OpenBLAS](#OpenBLAS)
* [DebugNodeInputsOutputs](#DebugNodeInputsOutputs)

**Architectures**
* [x86](#x86)
* [ARM](#ARM)
* [Android](#Android)

---



## Execution Providers

### CUDA
#### Pre-Requisites
* Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn)
  * ONNX Runtime is built and tested with CUDA 10.0 and cuDNN 7.6 using the Visual Studio 2017 14.11 toolset (i.e. Visual Studio 2017 v15.3). CUDA versions from 9.1 up to 10.1, and cuDNN versions from 7.1 up to 7.4 should also work with Visual Studio 2017.
  * The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home parameter`
  * The path to the cuDNN installation (include the `cuda` folder in the path) must be provided via the cuDNN_PATH environment variable, or `--cudnn_home parameter`. The cuDNN path should contain `bin`, `include` and `lib` directories.
  * The path to the cuDNN bin directory must be added to the PATH environment variable so that cudnn64_7.dll is found.

#### Build Instructions
##### Windows

```
.\build.bat --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path>
```

##### Linux
```
./build.sh --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path>
```

A Dockerfile is available [here](./dockerfiles#cuda).


#### Notes
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

### TensorRT

See more information on the TensorRT Execution Provider [here](./docs/execution_providers/TensorRT-ExecutionProvider.md).

#### Pre-Requisites
* Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn)
   * The TensorRT execution provider for ONNX Runtime is built and tested with CUDA 10.2 and cuDNN 7.6.5.
   * The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home parameter`. The CUDA path should contain `bin`, `include` and `lib` directories.
   * The path to the CUDA `bin` directory must be added to the PATH environment variable so that `nvcc` is found.
   * The path to the cuDNN installation (path to folder that contains libcudnn.so) must be provided via the cuDNN_PATH environment variable, or `--cudnn_home parameter`.
 * Install [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download)
   * The TensorRT execution provider for ONNX Runtime is built on TensorRT 7.x and is tested with TensorRT 7.0.0.11.
   * The path to TensorRT installation must be provided via the `--tensorrt_home parameter`.

#### Build Instructions
##### Windows

```
.\build.bat --cudnn_home <path to cuDNN home> --cuda_home <path to CUDA home> --use_tensorrt --tensorrt_home <path to TensorRT home>
```

##### Linux

```
./build.sh --cudnn_home <path to cuDNN e.g. /usr/lib/x86_64-linux-gnu/> --cuda_home <path to folder for CUDA e.g. /usr/local/cuda> --use_tensorrt --tensorrt_home <path to TensorRT home>
```

Dockerfile instructions are available [here](./dockerfiles#tensorrt)

#### Jetson (ARM64 Builds)

See [instructions](https://github.com/microsoft/onnxruntime/issues/2684#issuecomment-568548387) for additional information and tips related to building Onnxruntime with TensorRT Execution Provider on Jetson platforms (TX1/TX2, Nano)

---

### DNNL and MKLML
See more information on DNNL and MKL-ML [here](./docs/execution_providers/DNNL-ExecutionProvider.md).

#### Build Instructions
##### Linux
```
./build.sh --use_dnnl
```
---


### nGraph
See more information on the nGraph Execution Provider [here](./docs/execution_providers/nGraph-ExecutionProvider.md).

#### Build Instructions
#### Windows
```
.\build.bat --use_ngraph
```

##### Linux
```
./build.sh --use_ngraph
```

---

### OpenVINO
See more information on the OpenVINO Execution Provider [here](./docs/execution_providers/OpenVINO-ExecutionProvider.md).

#### Pre-Requisites
* Install the OpenVINO release along with its dependencies: [Windows]([https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit), [Linux](https://software.intel.com/en-us/openvino-toolkit).
   * For Linux, currently supports and is validated on OpenVINO 2019 R3.1
   * For Windows, download the 2019 R3.1 Windows Installer.
* Install the model optimizer prerequisites for ONNX by running:
   * Windows: `<openvino_install_dir>/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_onnx.bat`
   * Linux: `<openvino_install_dir>/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_onnx.sh`
* Initialize the OpenVINO environment by running the setupvars in `\<openvino\_install\_directory\>\/bin` using `setupvars.bat` (Windows) or `source setupvars.sh` (Linux)
   * To configure Intel<sup>®</sup> Processor Graphics(GPU) please follow these instructions: [Windows](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_installing_openvino_windows.html#Install-GPU), [Linux](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps)
   * To configure Intel<sup>®</sup> Movidius<sup>TM</sup> USB, please follow this getting started guide: [Windows](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_installing_openvino_windows.html#usb-myriad), [Linux](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_installing_openvino_linux.html#additional-NCS-steps)
   * To configure Intel<sup>®</sup> Vision Accelerator Design based on 8 Movidius<sup>TM</sup> MyriadX VPUs, please follow this configuration guide: [Windows](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_installing_openvino_windows.html#hddl-myriad), [Linux](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_installing_openvino_linux.html#install-VPU)
   * To configure Intel<sup>®</sup> Vision Accelerator Design with an Intel<sup>®</sup> Arria<sup>®</sup> 10 FPGA, please follow this configuration guide: [Linux](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_VisionAcceleratorFPGA_Configure_2019R3.html)


#### Build Instructions
##### Windows
```
.\build.bat --config RelWithDebInfo --use_openvino <hardware_option>
```
*Note: The default Windows CMake Generator is Visual Studio 2017, but you can also use the newer Visual Studio 2019 by passing `--cmake_generator "Visual Studio 16 2019"` to `.\build.bat`*

##### Linux
```
./build.sh --config RelWithDebInfo --use_openvino <hardware_option>
```

   <code>--use_openvino</code>: Builds the OpenVINO Execution Provider in ONNX Runtime.

  <code>--build_server</code>: Using this flag in addition to --use_openvino builds the OpenVINO Execution Provider with ONNX Runtime Server.

* `<hardware_option>`: Specifies the hardware target for building OpenVINO Execution Provider. Below are the options for different Intel target devices.

| Hardware Option | Target Device |
| --------------- | ------------------------|
| <code>CPU_FP32</code> | Intel<sup>®</sup> CPUs |
| <code>GPU_FP32</code> | Intel<sup>®</sup> Integrated Graphics |
| <code>GPU_FP16</code> | Intel<sup>®</sup> Integrated Graphics with FP16 quantization of models |
| <code>MYRIAD_FP16</code> | Intel<sup>®</sup> Movidius<sup>TM</sup> USB sticks | 
| <code>VAD-M_FP16</code> | Intel<sup>®</sup> Vision Accelerator Design based on 8 Movidius<sup>TM</sup> MyriadX VPUs |
| <code>VAD-F_FP32</code> | Intel<sup>®</sup> Vision Accelerator Design with an Intel<sup>®</sup> Arria<sup>®</sup> 10 FPGA |

For more information on OpenVINO Execution Provider&#39;s ONNX Layer support, Topology support, and Intel hardware enabled, please refer to the document [OpenVINO-ExecutionProvider.md](./docs/execution_providers/OpenVINO-ExecutionProvider.md) in <code>$onnxruntime_root/docs/execution_providers</code>

---

### Android NNAPI

See more information on the NNAPI Execution Provider [here](./docs/execution_providers/NNAPI-ExecutionProvider.md).

#### Pre-Requisites

To build ONNX Runtime with the NN API EP, first install Android NDK (see [Android Build instructions](#android))

#### Build Instructions

The basic build commands are below. There are also some other parameters for building the Android version. See [Android Build instructions](#android) for more details.

##### Cross compiling on Windows

```bash
./build.bat --android --android_sdk_path <android sdk path> --android_ndk_path <android ndk path> --dnnlibrary
```

##### Cross compiling on Linux

```bash
./build.sh --android --android_sdk_path <android sdk path> --android_ndk_path <android ndk path> --dnnlibrary
```

---

### NUPHAR
See more information on the Nuphar Execution Provider [here](./docs/execution_providers/Nuphar-ExecutionProvider.md).

#### Pre-Requisites
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

#### Build Instructions
##### Windows
```
.\build.bat --use_tvm --use_llvm --llvm_path=\llvm\install\path\lib\cmake\llvm --use_mklml --use_nuphar --build_shared_lib --build_csharp --enable_pybind --config=Release
```

* These instructions build the release flavor. The Debug build of LLVM would be needed to build with the Debug flavor of ONNX Runtime.

##### Linux:
```
./build.sh --use_tvm --use_llvm --llvm_path=/llvm/install/path/lib/cmake/llvm --use_mklml --use_nuphar --build_shared_lib --build_csharp --enable_pybind --config=Release
```

Dockerfile instructions are available [here](./dockerfiles#nuphar-public-preview)

---

### DirectML
See more information on the DirectML execution provider [here](./docs/execution_providers/DirectML-ExecutionProvider.md).
#### Windows
```
.\build.bat --use_dml
```
#### Notes
The DirectML execution provider supports building for both x64 and x86 architectures. DirectML is only supported on Windows.

---

### ARM Compute Library
See more information on the ACL Execution Provider [here](./docs/execution_providers/ACL-ExecutionProvider.md).

#### Prerequisites
* Supported backend: i.MX8QM Armv8 CPUs
* Supported BSP: i.MX8QM BSP
  * Install i.MX8QM BSP: `source fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4*.sh`
* Set up the build environment
```
source /opt/fsl-imx-xwayland/4.*/environment-setup-aarch64-poky-linux
alias cmake="/usr/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
```
* See [Build ARM](#ARM) below for information on building for ARM devices

#### Build Instructions

1. Configure ONNX Runtime with ACL support:
```
cmake ../onnxruntime-arm-upstream/cmake -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc -Donnxruntime_RUN_ONNX_TESTS=OFF -Donnxruntime_GENERATE_TEST_REPORTS=ON -Donnxruntime_DEV_MODE=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Donnxruntime_USE_CUDA=OFF -Donnxruntime_USE_NSYNC=OFF -Donnxruntime_CUDNN_HOME= -Donnxruntime_USE_JEMALLOC=OFF -Donnxruntime_ENABLE_PYTHON=OFF -Donnxruntime_BUILD_CSHARP=OFF -Donnxruntime_BUILD_SHARED_LIB=ON -Donnxruntime_USE_EIGEN_FOR_BLAS=ON -Donnxruntime_USE_OPENBLAS=OFF -Donnxruntime_USE_ACL=ON -Donnxruntime_USE_DNNL=OFF -Donnxruntime_USE_MKLML=OFF -Donnxruntime_USE_OPENMP=ON -Donnxruntime_USE_TVM=OFF -Donnxruntime_USE_LLVM=OFF -Donnxruntime_ENABLE_MICROSOFT_INTERNAL=OFF -Donnxruntime_USE_BRAINSLICE=OFF -Donnxruntime_USE_NUPHAR=OFF -Donnxruntime_USE_EIGEN_THREADPOOL=OFF -Donnxruntime_BUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
The ```-Donnxruntime_USE_ACL=ON``` option will use, by default, the 19.05 version of the Arm Compute Library. To set the right version you can use:
```-Donnxruntime_USE_ACL_1902=ON```, ```-Donnxruntime_USE_ACL_1905=ON``` or ```-Donnxruntime_USE_ACL_1908=ON```;

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

---

## Options
### OpenMP
#### Build Instructions
##### Windows
```
.\build.bat --use_openmp
```

##### Linux
```
./build.sh --use_openmp

```

---

### OpenBLAS
#### Pre-Requisites
* OpenBLAS
   * Windows: See build instructions [here](https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio#build-openblas-for-universal-windows-platform)
   * Linux: Install the libopenblas-dev package `sudo apt-get install libopenblas-dev`

#### Build Instructions
##### Windows
```
.\build.bat --use_openblas
```

##### Linux
```
./build.sh --use_openblas
```

---

### DebugNodeInputsOutputs
OnnxRuntime supports build options for enabling debugging of intermediate tensor shapes and data.
#### Build Instructions
##### Set onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
Dump tensor input/output shapes for all nodes to stdout.
```
# Linux
./build.sh --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
# Windows
.\build.bat --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
```
##### Set onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=2
Dump tensor input/output shapes and output data for all nodes to stdout.
```
# Linux
./build.sh --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=2
# Windows
.\build.bat --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=2
```
##### Set onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=0
To disable this functionality after previously enabling, set onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=0 or delete CMakeCache.txt.

---

## Architectures
### x86
#### Build Intsructions
##### Windows
* add `--x86` argument when launching `.\build.bat`

##### Linux
* Must be built on a x86 OS
* add --x86 argument to build.sh

---

### ARM
We have experimental support for Linux ARM builds. Windows on ARM is well tested.

#### Cross compiling for ARM with Docker (Linux/Windows - FASTER, RECOMMENDED)
This method allows you to compile using a desktop or cloud VM. This is much faster than compiling natively and avoids out-of-memory issues that may be encountered when on lower-powered ARM devices. The resulting ONNX Runtime Python wheel (.whl) file is then deployed to an ARM device where it can be invoked in Python 3 scripts.

See the instructions for the the Dockerfile [here](./dockerfiles/README.md#arm-32v7).

#### Cross compiling on Linux (without Docker)
1. Get the corresponding toolchain. For example, if your device is Raspberry Pi and the device os is Ubuntu 16.04, you may use gcc-linaro-6.3.1 from [https://releases.linaro.org/components/toolchain/binaries](https://releases.linaro.org/components/toolchain/binaries)
2. Setup env vars
    ```bash
       export PATH=/opt/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin:$PATH
       export CC=arm-linux-gnueabihf-gcc
       export CXX=arm-linux-gnueabihf-g++
    ```
3. Get a pre-compiled protoc:

   You may get it from https://github.com/protocolbuffers/protobuf/releases/download/v3.11.2/protoc-3.11.2-linux-x86_64.zip . Please unzip it after downloading.
4. (optional) Setup sysroot for enabling python extension. (TODO: will add details later)
5. Save the following content as tool.cmake
    ```
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR arm)
    set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-c++)
    set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
    ```
6. Append `-DONNX_CUSTOM_PROTOC_EXECUTABLE=/path/to/protoc -DCMAKE_TOOLCHAIN_FILE=path/to/tool.cmake` to your cmake args, run cmake and make to build it.

#### Native compiling on Linux ARM device (SLOWER)
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
wget https://cmake.org/files/v3.13/cmake-3.13.5.tar.gz;
tar zxf cmake-3.13.5.tar.gz

cd /code/cmake-3.13.5
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

#### Cross compiling on Windows
**Using Visual C++ compilers**
1. Download and install Visual C++ compilers and libraries for ARM(64).
   If you have Visual Studio installed, please use the Visual Studio Installer (look under the section `Individual components` after choosing to `modify` Visual Studio) to download and install the corresponding ARM(64) compilers and libraries.

2. Use `.\build.bat` and specify `--arm` or `--arm64` as the build option to start building. Preferably use `Developer Command Prompt for VS` or make sure all the installed cross-compilers are findable from the command prompt being used to build using the PATH environmant variable.

---

### Android

#### Pre-Requisites

Install Android NDK in Android Studio or https://developer.android.com/ndk/downloads

#### Build Instructions

##### Cross compiling on Windows

```bash
./build.bat --android --android_sdk_path <android sdk path> --android_ndk_path <android ndk path> --android_abi <android abi, e.g., arm64-v8a (default) or armeabi-v7a> --android_api <android api level, e.g., 27 (default)>
```

##### Cross compiling on Linux

```bash
./build.sh --android --android_sdk_path <android sdk path> --android_ndk_path <android ndk path> --android_abi <android abi, e.g., arm64-v8a (default) or armeabi-v7a> --android_api <android api level, e.g., 27 (default)>
```

Android Archive (AAR) files, which can be imported directly in Android Studio, will be generated in your_build_dir/java/build/outputs/aar.

If you want to use NNAPI Execution Provider on Android, see [docs/execution_providers/NNAPI-ExecutionProvider.md](/docs/execution_providers/NNAPI-ExecutionProvider.md).
