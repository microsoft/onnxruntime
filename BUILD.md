# Build ONNX Runtime

## Supported architectures

|           | x86_32       | x86_64       | ARM32v7      | ARM64        |
|-----------|:------------:|:------------:|:------------:|:------------:|
|Windows    | YES          | YES          |  YES         | YES          |
|Linux      | YES          | YES          |  YES         | YES          |
|Mac OS X   | NO           | YES          |  NO          | NO           |

## Supported dev environments

| OS          | Supports CPU | Supports GPU| Notes                              |
|-------------|:------------:|:------------:|------------------------------------|
|Windows 10   | YES          | YES         |Must use VS 2017 or the latest VS2015|
|Windows 10 <br/> Subsystem for Linux | YES         | NO        |         |
|Ubuntu 16.x  | YES          | YES         | Also supported on ARM32v7 (experimental) |

* Red Hat Enterprise Linux and CentOS are not supported.
* Other version of Ubuntu might work but we don't support them officially.
* GCC 4.x and below are not supported.

OS/Compiler Matrix:

| OS/Compiler | Supports VC  | Supports GCC     |
|-------------|:------------:|:----------------:|
|Windows 10   | YES          | Not tested       |
|Linux        | NO           | YES(gcc>=5.0)    |

ONNX Runtime python binding only supports Python 3.5, 3.6 and 3.7.

## Build
1. Checkout the source tree:
   ```
   git clone --recursive https://github.com/Microsoft/onnxruntime
   cd onnxruntime
   ```
2. Install cmake-3.13 or better from https://cmake.org/download/.
3. (optional) Install protobuf 3.6.1 from source code (cmake/external/protobuf). CMake flag protobuf\_BUILD\_SHARED\_LIBS must be turned off. After the installation, you should have the 'protoc' executable in your PATH.
4. (optional) Install onnx from source code (cmake/external/onnx)
    ```
    export ONNX_ML=1
    python3 setup.py bdist_wheel
    pip3 install --upgrade dist/*.whl
    ```
5. Run `./build.sh --config RelWithDebInfo --build_wheel` for Linux (or `build.bat --config RelWithDebInfo --build_wheel` for Windows)

The build script runs all unit tests by default (for native builds and skips tests by default for cross-compiled builds).

The complete list of build options can be found by running `./build.sh (or ./build.bat) --help`

## Build x86
1. For Windows, just add --x86 argument when launching build.bat
2. For Linux, it must be built out of a x86 os, --x86 argument also needs be specified to build.sh

## Build ONNX Runtime Server on Linux

1. In the ONNX Runtime root folder, run `./build.sh --config RelWithDebInfo --build_server  --use_openmp --parallel`

## Build/Test Flavors for CI

### CI Build Environments

| Build Job Name     | Environment         | Dependency                      | Test Coverage            | Scripts                                  |
|--------------------|---------------------|---------------------------------|--------------------------|------------------------------------------|
| Linux_CI_Dev       | Ubuntu 16.04        | python=3.5                      | Unit tests; ONNXModelZoo | [script](tools/ci_build/github/linux/run_build.sh) |
| Linux_CI_GPU_Dev   | Ubuntu 16.04        | python=3.5; nvidia-docker       | Unit tests; ONNXModelZoo | [script](tools/ci_build/github/linux/run_build.sh) |
| Windows_CI_Dev     | Windows Server 2016 | python=3.5                      | Unit tests; ONNXModelZoo | [script](build.bat)                                |
| Windows_CI_GPU_Dev | Windows Server 2016 | cuda=9.1; cudnn=7.1; python=3.5 | Unit tests; ONNXModelZoo | [script](build.bat)                                |

## Additional Build Flavors
The complete list of build flavors can be seen by running `./build.sh --help` or `./build.bat --help`. Here are some common flavors.

### Windows CUDA Build
ONNX Runtime supports CUDA builds. You will need to download and install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [CUDNN](https://developer.nvidia.com/cudnn).

ONNX Runtime is built and tested with CUDA 9.1 and CUDNN 7.1 using the Visual Studio 2017 14.11 toolset (i.e. Visual Studio 2017 v15.3).
CUDA versions from 9.1 up to 10.0, and CUDNN versions from 7.1 up to 7.4 should also work with Visual Studio 2017.

 - The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home parameter`.
 - The path to the CUDNN installation (include the `cuda` folder in the path) must be provided via the CUDNN_PATH environment variable, or `--cudnn_home parameter`. The CUDNN path should contain `bin`, `include` and `lib` directories.
 - The path to the CUDNN bin directory must be added to the PATH environment variable so that cudnn64_7.dll is found.

You can build with:

```
./build.sh --use_cuda --cudnn_home /usr --cuda_home /usr/local/cuda (Linux)
./build.bat --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path> (Windows)
```

Depending on compatibility between the CUDA, CUDNN, and Visual Studio 2017 versions you are using, you may need to explicitly install an earlier version of the MSVC toolset.
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

### MKL-DNN/MKLML
To build ONNX Runtime with MKL-DNN support, build it with `./build.sh --use_mkldnn`
To build ONNX Runtime using MKL-DNN built with dependency on MKL small libraries, build it with `./build.sh --use_mkldnn --use_mklml`

### TensorRT
ONNX Runtime supports the TensorRT execution provider (released as preview). You will need to download and install [CUDA](https://developer.nvidia.com/cuda-toolkit), [CUDNN](https://developer.nvidia.com/cudnn) and [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download).

The TensorRT execution provider for ONNX Runtime is built and tested with CUDA 9.0/CUDA 10.0, CUDNN 7.1 and TensorRT 5.0.2.6.

 - The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home parameter`. The CUDA path should contain `bin`, `include` and `lib` directories.
 - The path to the CUDA `bin` directory must be added to the PATH environment variable so that `nvcc` is found.
 - The path to the CUDNN installation (path to folder that contains libcudnn.so) must be provided via the CUDNN_PATH environment variable, or `--cudnn_home parameter`.
- The path to TensorRT installation must be provided via the `--tensorrt_home parameter`.

You can build from source on Linux by using the following `cmd` from the onnxruntime directory:

```
./build.sh --cudnn_home <path to CUDNN e.g. /usr/lib/x86_64-linux-gnu/> --cuda_home <path to folder for CUDA e.g. /usr/local/cuda> --use_tensorrt --tensorrt_home <path to TensorRT home> (Linux)

```


### OpenBLAS
#### Windows
Instructions how to build OpenBLAS for windows can be found here https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio#build-openblas-for-universal-windows-platform.

Once you have the OpenBLAS binaries, build ONNX Runtime with `./build.bat --use_openblas`

#### Linux
For Linux (e.g. Ubuntu 16.04), install libopenblas-dev package
`sudo apt-get install libopenblas-dev` and build with `./build.sh --use_openblas`

### OpenMP
```
./build.sh --use_openmp (for Linux)
./build.bat --use_openmp (for Windows)
```

### Build with Docker on Linux
Install Docker: `https://docs.docker.com/install/`

#### CPU
```
cd tools/ci_build/github/linux/docker
docker build -t onnxruntime_dev --build-arg OS_VERSION=16.04 -f Dockerfile.ubuntu .
docker run --rm -it onnxruntime_dev /bin/bash
```

#### GPU
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

## ARM Builds
We have experimental support for Linux ARM builds. Windows on ARM is well tested.

### Cross compiling for ARM with Docker (Linux/Windows - FASTER, RECOMMENDED)
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

### Cross compiling on Linux (without Docker)
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


### Native compiling on Linux ARM device (SLOWER)
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

### Cross compiling on Windows
#### Using Visual C++ compilers
1. Download and install Visual C++ compilers and libraries for ARM(64).
   If you have Visual Studio installed, please use the Visual Studio Installer (look under the section `Individual components` after choosing to `modify` Visual Studio) to download and install the corresponding ARM(64) compilers and libraries.

2. Use `build.bat` and specify `--arm` or `--arm64` as the build option to start building. Preferably use `Developer Command Prompt for VS` or make sure all the installed cross-compilers are findable from the command prompt being used to build using the PATH environmant variable.

### Using other compilers
(TODO)
