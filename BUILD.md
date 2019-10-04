# Building ONNX Runtime - Getting Started
*Dockerfiles are available [here](https://github.com/microsoft/onnxruntime/tree/master/tools/ci_build/github/linux/docker) to help you get started.*

*Pre-built packages are available at the locations indicated [here](https://github.com/microsoft/onnxruntime#official-builds).*

## To build the baseline CPU version of ONNX Runtime from source:
1. Checkout the source tree:
   ```
   git clone --recursive https://github.com/Microsoft/onnxruntime
   cd onnxruntime
   ```
2. Install cmake-3.13 or better from https://cmake.org/download/.

**On Windows:**

3. (optional) Install protobuf 3.6.1 from source code (cmake/external/protobuf). CMake flag protobuf\_BUILD\_SHARED\_LIBS must be turned OFF. After the installation, you should have the 'protoc' executable in your PATH.
4. (optional) Install onnx from source code (cmake/external/onnx)
    ```
    export ONNX_ML=1
    python3 setup.py bdist_wheel
    pip3 install --upgrade dist/*.whl
    ```
5. Run `build.bat --config RelWithDebInfo --build_shared_lib --parallel`.

*Note: The default Windows CMake Generator is Visual Studio 2017, but you can also use the newer Visual Studio 2019 by passing `--cmake_generator "Visual Studio 16 2019"` to build.bat.*

**On Linux:**

3. (optional) Install protobuf 3.6.1 from source code (cmake/external/protobuf). CMake flag protobuf\_BUILD\_SHARED\_LIBS must be turned ON. After the installation, you should have the 'protoc' executable in your PATH. It is recommended to run `ldconfig` to make sure protobuf libraries are found.
4. If you installed your protobuf in a non standard location it would be helpful to set the following env var:`export CMAKE_ARGS="-DONNX_CUSTOM_PROTOC_EXECUTABLE=full path to protoc"` so ONNX build can find it. Also run `ldconfig <protobuf lib folder path>` so the linker can find protobuf libraries.
5. (optional) Install onnx from source code (cmake/external/onnx)
    ```
    export ONNX_ML=1
    python3 setup.py bdist_wheel
    pip3 install --upgrade dist/*.whl
    ```
6. Run `./build.sh --config RelWithDebInfo --build_shared_lib --parallel`.

The build script runs all unit tests by default (for native builds and skips tests by default for cross-compiled builds).

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

* Red Hat Enterprise Linux and CentOS are not supported.
* Other version of Ubuntu might work but we don't support them officially.
* GCC 4.x and below are not supported.

### OS/Compiler Matrix:

| OS/Compiler | Supports VC  | Supports GCC     |
|-------------|:------------:|:----------------:|
|Windows 10   | YES          | Not tested       |
|Linux        | NO           | YES(gcc>=5.0)    |

ONNX Runtime Python bindings support Python 3.5, 3.6 and 3.7.

---

# Additional Build Instructions
The complete list of build options can be found by running `./build.sh (or ./build.bat) --help`

* [Docker on Linux](#Docker-on-Linux)
* [ONNX Runtime Server (Linux)](#Build-ONNX-Runtime-Server-on-Linux)

**Execution Providers**
* [NVIDIA CUDA](#CUDA)
* [NVIDIA TensorRT](#TensorRT)
* [Intel MKL-DNN/MKL-ML](#MKLDNN-and-MKLML)
* [Intel nGraph](#nGraph)
* [Intel OpenVINO](#openvino)
* [Android NNAPI](#Android)
* [Nuphar](#Nuphar)

**Options**
* [OpenMP](#OpenMP)
* [OpenBLAS](#OpenBLAS)

**Architectures**
* [x86](#x86)
* [ARM](#ARM)

---
## Docker on Linux
Install Docker: `https://docs.docker.com/install/`

**CPU**
```
cd tools/ci_build/github/linux/docker
docker build -t onnxruntime_dev --build-arg OS_VERSION=16.04 -f Dockerfile.ubuntu .
docker run --rm -it onnxruntime_dev /bin/bash
```

**GPU**
If you need GPU support, please also install:
1. nvidia driver. Before doing this please add `nomodeset rd.driver.blacklist=nouveau` to your linux [kernel boot parameters](https://www.kernel.org/doc/html/v4.17/admin-guide/kernel-parameters.html).
2. nvidia-docker2: [Install doc](`https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)`)

To test if your nvidia-docker works:
```
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```

Then build a docker image. We provided a sample for use:
```
cd tools/ci_build/github/linux/docker
docker build -t cuda_dev -f Dockerfile.ubuntu_gpu .
```

Then run it
```
./tools/ci_build/github/linux/run_dockerbuild.sh
```

---

## Build ONNX Runtime Server on Linux
Read more about ONNX Runtime Server [here](https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Server_Usage.md)
1. ONNX Runtime server (and only the server) requires you to have Go installed to build, due to building BoringSSL.
    See https://golang.org/doc/install for installation instructions.
2. In the ONNX Runtime root folder, run `./build.sh --config RelWithDebInfo --build_server  --use_openmp --parallel`
3. ONNX Runtime Server supports sending log to [rsyslog](https://www.rsyslog.com/) daemon. To enable it, please build with an additional parameter: `--cmake_extra_defines onnxruntime_USE_SYSLOG=1`. The build command will look like this: `./build.sh --config RelWithDebInfo --build_server  --use_openmp --parallel --cmake_extra_defines onnxruntime_USE_SYSLOG=1`

---

## Execution Providers

### CUDA
For Linux, please use [this Dockerfile](https://github.com/microsoft/onnxruntime/blob/master/tools/ci_build/github/linux/docker/Dockerfile.ubuntu_gpu) and refer to instructions above for [building with Docker on Linux](#Docker-on-Linux)

ONNX Runtime supports CUDA builds. You will need to download and install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn).

ONNX Runtime is built and tested with CUDA 10.0 and cuDNN 7.3 using the Visual Studio 2017 14.11 toolset (i.e. Visual Studio 2017 v15.3).
CUDA versions from 9.1 up to 10.1, and cuDNN versions from 7.1 up to 7.4 should also work with Visual Studio 2017.

 - The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home parameter`.
 - The path to the cuDNN installation (include the `cuda` folder in the path) must be provided via the cuDNN_PATH environment variable, or `--cudnn_home parameter`. The cuDNN path should contain `bin`, `include` and `lib` directories.
 - The path to the cuDNN bin directory must be added to the PATH environment variable so that cudnn64_7.dll is found.

You can build with:

```
./build.sh --use_cuda --cudnn_home /usr --cuda_home /usr/local/cuda (Linux)
./build.bat --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path> (Windows)
```

Depending on compatibility between the CUDA, cuDNN, and Visual Studio 2017 versions you are using, you may need to explicitly install an earlier version of the MSVC toolset.
- CUDA 10.0 is known to work with toolsets from 14.11 up to 14.16 (Visual Studio 2017 15.9), and should continue to work with future Visual Studio versions
  - https://devblogs.microsoft.com/cppblog/cuda-10-is-now-available-with-support-for-the-latest-visual-studio-2017-versions/
- CUDA 9.2 is known to work with the 14.11 MSVC toolset (Visual Studio 15.3 and 15.4)

To install the 14.11 MSVC toolset, see <https://blogs.msdn.microsoft.com/vcblog/2017/11/15/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2017/>

To use the 14.11 toolset with a later version of Visual Studio 2017 you have two options:

1. Setup the Visual Studio environment variables to point to the 14.11 toolset by running vcvarsall.bat, prior to running the build script
   - e.g.  if you have VS2017 Enterprise, an x64 build would use the following command
`"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.11`
   - For convenience, build.amd64.1411.bat will do this and can be used in the same way as build.bat.
     - e.g.` .\build.amd64.1411.bat --use_cuda`

2. Alternatively if you have CMake 3.12 or later you can specify the toolset version via the `--msvc_toolset` build script parameter.
   - e.g. `.\build.bat --msvc_toolset 14.11`

_Side note: If you have multiple versions of CUDA installed on a Windows machine and are building with Visual Studio, CMake will use the build files for the highest version of CUDA it finds in the BuildCustomization folder.
e.g. C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\IDE\VC\VCTargets\BuildCustomizations\.
If you want to build with an earlier version, you must temporarily remove the 'CUDA x.y.*' files for later versions from this directory._

---

### TensorRT
ONNX Runtime supports the TensorRT execution provider (released as preview). You will need to download and install [CUDA](https://developer.nvidia.com/cuda-toolkit), [cuDNN](https://developer.nvidia.com/cudnn) and [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download).

The TensorRT execution provider for ONNX Runtime is built and tested with CUDA 9.0/CUDA 10.0, cuDNN 7.1 and TensorRT 5.0.2.6.

 - The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home parameter`. The CUDA path should contain `bin`, `include` and `lib` directories.
 - The path to the CUDA `bin` directory must be added to the PATH environment variable so that `nvcc` is found.
 - The path to the cuDNN installation (path to folder that contains libcudnn.so) must be provided via the cuDNN_PATH environment variable, or `--cudnn_home parameter`.
- The path to TensorRT installation must be provided via the `--tensorrt_home parameter`.

You can build from source on Linux by using the following `cmd` from the onnxruntime directory:

```
./build.sh --cudnn_home <path to cuDNN e.g. /usr/lib/x86_64-linux-gnu/> --cuda_home <path to folder for CUDA e.g. /usr/local/cuda> --use_tensorrt --tensorrt_home <path to TensorRT home> (Linux)
```

---

### MKLDNN and MKLML
To build ONNX Runtime with MKL-DNN support, build it with `./build.sh --use_mkldnn`
To build ONNX Runtime using MKL-DNN built with dependency on MKL small libraries, build it with `./build.sh --use_mkldnn --use_mklml`

---

### nGraph
ONNX runtime with nGraph as an execution provider (released as preview) can be built on Linux as follows : `./build.sh --use_ngraph`.  Similarly, on Windows use `.\build.bat --use_ngraph`

---

### OpenVINO

ONNX Runtime supports OpenVINO Execution Provider to enable deep learning inference using Intel<sup>®</sup> OpenVINO<sup>TM</sup> Toolkit. This execution provider supports several Intel hardware device types - CPU, integrated GPU, Intel<sup>®</sup> Movidius<sup>TM</sup> VPUs and Intel<sup>®</sup> Vision accelerator Design with 8 Intel Movidius<sup>TM</sup> MyriadX VPUs.

The OpenVINO Execution Provider can be built using the following commands:

- Currently supports and validated on two versions of OpenVINO: OpenVINO 2018 R5.0.1 and OpenVINO 2019 R1.1(Recommended). Install the OpenVINO release along with its dependencies from ([https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)).For windows, please download 2019 R1.1 windows installer

- Install the model optimizer prerequisites for ONNX by running

   For Linux:

    <code><openvino_install_dir>/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_onnx.sh</code>

    For Windows:

    <code><openvino_install_dir>/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_onnx.bat</code>

- Initialize the OpenVINO environment by running the setupvars in <code>\<openvino\_install\_directory\>\/bin</code> using the below command:

   <code>source setupvars.sh (Linux)</code>

   <code>setupvars.bat (Windows)</code>

- To configure Intel<sup>®</sup> Processor Graphics(GPU), please follow the installation steps from    (https://docs.openvinotoolkit.org/2019_R1.1/_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps  (Linux))
 (https://docs.openvinotoolkit.org/2019_R1.1/_docs_install_guides_installing_openvino_windows.html#Install-GPU (Windows))

- To configure Intel<sup>®</sup> Movidius<sup>TM</sup> USB, please follow the getting started guide from (https://docs.openvinotoolkit.org/2019_R1.1/_docs_install_guides_installing_openvino_linux.html#additional-NCS-steps (Linux))
(https://docs.openvinotoolkit.org/2019_R1.1/_docs_install_guides_installing_openvino_windows.html#usb-myriad (Windows))

- To configure Intel<sup>®</sup> Vision Accelerator Design based on 8 Movidius<sup>TM</sup> MyriadX VPUs, please follow the configuration guide from (https://docs.openvinotoolkit.org/2019_R1.1/_docs_install_guides_installing_openvino_linux.html#install-VPU (Linux))
(https://docs.openvinotoolkit.org/2019_R1.1/_docs_install_guides_installing_openvino_windows.html#hddl-myriad (Windows))

- To configure Intel<sup>®</sup> Vision Accelerator Design with an Intel<sup>®</sup> Arria<sup>®</sup> 10 FPGA, please follow the configuration guide from (https://docs.openvinotoolkit.org/2019_R1.1/_docs_install_guides_VisionAcceleratorFPGA_Configure_2019R1.html)


- Build ONNX Runtime using the below command.

  For Linux:
  
   <code>./build.sh --config RelWithDebInfo --use_openvino <hardware_option>  </code>

  For Windows:
  
  <code> build.bat --config RelWithDebInfo  --use_openvino <hardware_option> </code>
 
   *Note: The default Windows CMake Generator is Visual Studio 2017, but you can also use the newer Visual Studio 2019 by passing `--cmake_generator "Visual Studio 16 2019"` to build.bat.*
 
   <code>--use_openvino</code>: Builds the OpenVINO Execution Provider in ONNX Runtime.

   <code><hardware_option></code>: Specifies the hardware target for building OpenVINO Execution Provider. Below are the options for different Intel target devices.

| Hardware Option | Target Device |
| --------------- | ------------------------|
| <code>CPU_FP32</code> | Intel<sup>®</sup> CPUs |
| <code>GPU_FP32</code> | Intel<sup>®</sup> Integrated Graphics |
| <code>GPU_FP16</code> | Intel<sup>®</sup> Integrated Graphics with FP16 quantization of models |
| <code>MYRIAD_FP16</code> | Intel<sup>®</sup> Movidius<sup>TM</sup> USB sticks | 
| <code>VAD-M_FP16</code> | Intel<sup>®</sup> Vision Accelerator Design based on 8 Movidius<sup>TM</sup> MyriadX VPUs |
| <code>VAD-F_FP32</code> | Intel<sup>®</sup> Vision Accelerator Design with an Intel<sup>®</sup> Arria<sup>®</sup> 10 FPGA |

For more information on OpenVINO Execution Provider&#39;s ONNX Layer support, Topology support, and Intel hardware enabled, please refer to the document OpenVINO-ExecutionProvider.md in <code>$onnxruntime_root/docs/execution_providers</code>

---

### Android

#### Cross compiling on Linux

1. Get Android NDK from https://developer.android.com/ndk/downloads. Please unzip it after downloading.

2. Get a pre-compiled protoc:

   You may get it from https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip. Please unzip it after downloading.

3. Denote the unzip destination in step 1 as $ANDROID_NDK, append `-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DONNX_CUSTOM_PROTOC_EXECUTABLE=path/to/protoc` to your cmake args, run cmake and make to build it.

Note: For 32-bit devices, replace `-DANDROID_ABI=arm64-v8a` to `-DANDROID_ABI=armeabi-v7a`.

---

### Nuphar
ONNX Runtime supports Nuphar execution provider (released as preview). It is an execution provider built on top of [TVM](https://github.com/dmlc/tvm) and [LLVM](https://llvm.org). Currently it targets to X64 CPU.

The Nuphar execution provider for ONNX Runtime is built and tested with LLVM 9.0.0. Because of TVM's requirement when building with LLVM, you need to build LLVM from source:

Windows with Visual Studio 2017: (Note here builds release flavor. Debug build of LLVM would be needed to build with Debug flavor of ONNX Runtime)
```
REM download llvm source code 9.0.0 and unzip to \llvm\source\path, then install to \llvm\install\path
cd \llvm\source\path
mkdir build
cd build
cmake .. -G "Visual Studio 15 2017 Win64" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_DIA_SDK=OFF
msbuild llvm.sln /maxcpucount /p:Configuration=Release /p:Platform=x64
cmake -DCMAKE_INSTALL_PREFIX=\llvm\install\path -DBUILD_TYPE=Release -P cmake_install.cmake
```

Note that following LLVM cmake patch is necessary to make the build work on Windows, Linux does not need to apply the patch.
The patch is to fix the linking warning LNK4199 caused by [LLVM commit](https://github.com/llvm-mirror/llvm/commit/148f823e4845c9a13faea62e3105abb80b39e4bc)

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

Linux:
```
# download llvm source code 9.0.0 and unzip to /llvm/source/path, then install to /llvm/install/path
cd /llvm/source/path
mkdir build
cd build
cmake .. -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE=Release
cmake --build.
cmake -DCMAKE_INSTALL_PREFIX=/llvm/install/path -DBUILD_TYPE=Release -P cmake_install.cmake
```

Then you can build from source by using following command from the onnxruntime directory:
Windows:
```
build.bat --use_tvm --use_llvm --llvm_path=\llvm\install\path\lib\cmake\llvm --use_mklml --use_nuphar --build_shared_lib --build_csharp --enable_pybind --config=Release
```

Linux:
```
./build.sh --use_tvm --use_llvm --llvm_path=/llvm/install/path/lib/cmake/llvm --use_mklml --use_nuphar --build_shared_lib --build_csharp --enable_pybind --config=Release
```

---

## Options
### OpenMP
```
./build.sh --use_openmp (for Linux)
./build.bat --use_openmp (for Windows)
```

---

### OpenBLAS
**Windows**
Instructions how to build OpenBLAS for windows can be found here https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio#build-openblas-for-universal-windows-platform.

Once you have the OpenBLAS binaries, build ONNX Runtime with `./build.bat --use_openblas`

**Linux**
For Linux (e.g. Ubuntu 16.04), install libopenblas-dev package
`sudo apt-get install libopenblas-dev` and build with `./build.sh --use_openblas`

---

## Architectures
### x86
 - For Windows, just add --x86 argument when launching build.bat
 - For Linux, it must be built out of a x86 os, --x86 argument also needs be specified to build.sh

---

### ARM
We have experimental support for Linux ARM builds. Windows on ARM is well tested.

#### Cross compiling for ARM with Docker (Linux/Windows - FASTER, RECOMMENDED)
This method allows you to compile using a desktop or cloud VM. This is much faster than compiling natively and avoids out-of-memory issues that may be encountered when on lower-powered ARM devices. The resulting ONNX Runtime Python wheel (.whl) file is then deployed to an ARM device where it can be invoked in Python 3 scripts.

The Dockerfile used in these instructions specifically targets Raspberry Pi 3/3+ running Raspbian Stretch. The same approach should work for other ARM devices, but may require some changes to the Dockerfile such as choosing a different base image (Line 0: `FROM ...`).

1. Install DockerCE on your development machine by following the instructions [here](https://docs.docker.com/install/)
2. Create an empty local directory
    ```bash
    mkdir onnx-build
    cd onnx-build
    ```
3. Save the Dockerfile to your new directory
    - [Dockerfile.arm32v7](https://github.com/Microsoft/onnxruntime/blob/master/dockerfiles/Dockerfile.arm32v7)
4. Run docker build

    This will build all the dependencies first, then build ONNX Runtime and its Python bindings. This will take several hours.
    ```bash
    docker build -t onnxruntime-arm32v7 -f Dockerfile.arm32v7 .
    ```
5. Note the full path of the `.whl` file

    - Reported at the end of the build, after the `# Build Output` line.
    - It should follow the format `onnxruntime-0.3.0-cp35-cp35m-linux_armv7l.whl`, but version number may have changed. You'll use this path to extract the wheel file later.
6. Check that the build succeeded

    Upon completion, you should see an image tagged `onnxruntime-arm32v7` in your list of docker images:
    ```bash
    docker images
    ```
7. Extract the Python wheel file from the docker image

    (Update the path/version of the `.whl` file with the one noted in step 5)
    ```bash
    docker create -ti --name onnxruntime_temp onnxruntime-arm32v7 bash
    docker cp onnxruntime_temp:/code/onnxruntime/build/Linux/MinSizeRel/dist/onnxruntime-0.3.0-cp35-cp35m-linux_armv7l.whl .
    docker rm -fv onnxruntime_temp
    ```
    This will save a copy of the wheel file, `onnxruntime-0.3.0-cp35-cp35m-linux_armv7l.whl`, to your working directory on your host machine.
8. Copy the wheel file (`onnxruntime-0.3.0-cp35-cp35m-linux_armv7l.whl`) to your Raspberry Pi or other ARM device
9. On device, install the ONNX Runtime wheel file
    ```bash
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip
    pip3 install numpy

    # Install ONNX Runtime
    # Important: Update path/version to match the name and location of your .whl file
    pip3 install onnxruntime-0.3.0-cp35-cp35m-linux_armv7l.whl
    ```
10. Test installation by following the instructions [here](https://microsoft.github.io/onnxruntime/)

#### Cross compiling on Linux (without Docker)
1. Get the corresponding toolchain. For example, if your device is Raspberry Pi and the device os is Ubuntu 16.04, you may use gcc-linaro-6.3.1 from [https://releases.linaro.org/components/toolchain/binaries](https://releases.linaro.org/components/toolchain/binaries)
2. Setup env vars
    ```bash
       export PATH=/opt/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin:$PATH
       export CC=arm-linux-gnueabihf-gcc
       export CXX=arm-linux-gnueabihf-g++
    ```
3. Get a pre-compiled protoc:

   You may get it from https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip . Please unzip it after downloading.
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
wget https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz;
tar zxf cmake-3.12.3.tar.gz

cd /code/cmake-3.12.3
./configure --system-curl
make
sudo make install

# Prepare onnxruntime Repo
cd /code
git clone --recursive https://github.com/Microsoft/onnxruntime

# Start the basic build
cd /code/onnxruntime
./build.sh --config MinSizeRel --arm --update --build

# Build Shared Library
./build.sh --config MinSizeRel --arm --build_shared_lib

# Build Python Bindings and Wheel
./build.sh --config MinSizeRel --arm --enable_pybind --build_wheel

# Build Output
ls -l /code/onnxruntime/build/Linux/MinSizeRel/*.so
ls -l /code/onnxruntime/build/Linux/MinSizeRel/dist/*.whl
```

#### Cross compiling on Windows
**Using Visual C++ compilers**
1. Download and install Visual C++ compilers and libraries for ARM(64).
   If you have Visual Studio installed, please use the Visual Studio Installer (look under the section `Individual components` after choosing to `modify` Visual Studio) to download and install the corresponding ARM(64) compilers and libraries.

2. Use `build.bat` and specify `--arm` or `--arm64` as the build option to start building. Preferably use `Developer Command Prompt for VS` or make sure all the installed cross-compilers are findable from the command prompt being used to build using the PATH environmant variable.

