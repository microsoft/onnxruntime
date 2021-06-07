---
title: Build for inferencing
parent: Build ORT
grand_parent: How to
nav_order: 1
---

# Build ONNX Runtime for inferencing
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## CPU
Basic CPU build

### Prerequisites
{: .no_toc }

* Checkout the source tree:
   ```
   git clone --recursive https://github.com/Microsoft/onnxruntime
   cd onnxruntime
   ```
* [Install](https://cmake.org/download/) cmake-3.18 or higher.


### Build Instructions
{: .no_toc }

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

#### macOS

By default, ORT is configured to be built for a minimum target macOS version of 10.12.
The shared library in the release Nuget(s) and the Python wheel may be installed on macOS versions of 10.12+.

If you would like to use [Xcode](https://developer.apple.com/xcode/) to build the onnxruntime for x86_64 macOS, please add the --user_xcode argument in the command line.

Without this flag, the cmake build generator will be Unix makefile by default.
Also, if you want to cross-compile for Apple Silicon in an Intel-based MacOS machine, please add the argument --osx_arch arm64 with cmake > 3.19. Note: unit tests will be skipped due to the incompatible CPU instruction set.

#### Notes

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

## Supported architectures and build environments

### Architectures
{: .no_toc }

|           | x86_32       | x86_64       | ARM32v7      | ARM64        |
|-----------|:------------:|:------------:|:------------:|:------------:|
|Windows    | YES          | YES          |  YES         | YES          |
|Linux      | YES          | YES          |  YES         | YES          |
|macOS      | NO           | YES          |  NO          | NO           |

### Environments
{: .no_toc }

| OS          | Supports CPU | Supports GPU| Notes                              |
|-------------|:------------:|:------------:|------------------------------------|
|Windows 10   | YES          | YES         | VS2019 through the latest VS2015 are supported |
|Windows 10 <br/> Subsystem for Linux | YES         | NO        |         |
|Ubuntu 16.x  | YES          | YES         | Also supported on ARM32v7 (experimental) |
|macOS        | YES          | NO         |    |

GCC 4.x and below are not supported.

### OS/Compiler Matrix
{: .no_toc }

| OS/Compiler | Supports VC  | Supports GCC     |  Supports Clang  |
|-------------|:------------:|:----------------:|:----------------:|
|Windows 10   | YES          | Not tested       | Not tested       |
|Linux        | NO           | YES(gcc>=8)      | Not tested       |
|macOS        | NO           | Not tested       | YES (Minimum version required not ascertained)|

---

## Common Build Instructions

|Description|Command|Additional details|
|-----------|-----------|-----------|
|**Basic build**|build.bat (Windows)<br>./build.sh (Linux)||
|**Release build**|--config Release|Release build. Other valid config values are RelWithDebInfo and Debug.|
|**Build using parallel processing**|--parallel|This is strongly recommended to speed up the build.|
|**Build Shared Library**|--build_shared_lib||
|**Enable Training support**|--enable_training||

### APIs and Language Bindings

|API|Command|Additional details|
|-----------|-----------|-----------|
|**Python**|--build_wheel||
|**C# and C Nuget packages**|--build_nuget|Builds C# bindings and creates nuget package. Implies `--build_shared_lib` <br> Detailed instructions can be found [below](#build-nuget-packages).|
|**WindowsML**|--use_winml<br>--use_dml<br>--build_shared_lib|WindowsML depends on DirectML and the OnnxRuntime shared library|
|**Java**|--build_java|Creates an onnxruntime4j.jar in the build directory, implies `--build_shared_lib`<br>Compiling the Java API requires [gradle](https://gradle.org) v6.1+ to be installed in addition to the usual requirements.|
|**Node.js**|--build_nodejs|Build Node.js binding. Implies `--build_shared_lib`|


#### Build Nuget packages

Currently only supported on Windows and Linux.

##### Prerequisites

* dotnet is required for building csharp bindings and creating managed nuget package. Follow the instructions [here](https://dotnet.microsoft.com/download) to download dotnet. Tested with versions 2.1 and 3.1.
* nuget.exe. Follow the instructions [here](https://docs.microsoft.com/en-us/nuget/install-nuget-client-tools#nugetexe-cli) to download nuget
  * On Windows, downloading nuget is straightforward and simply following the instructions above should work.
  * On Linux, nuget relies on Mono runtime and therefore this needs to be setup too. Above link has all the information to setup Mono and nuget. The instructions can directly be found [here](https://www.mono-project.com/docs/getting-started/install/). In some cases it is required to run `sudo apt-get install mono-complete` after installing mono.

##### Build Instructions
###### Windows
```
.\build.bat --build_nuget
```

###### Linux
```
./build.sh --build_nuget
```
Nuget packages are created under <native_build_dir>\nuget-artifacts

---

## Other build options

### Reduced Operator Kernel Build
Reduced Operator Kernel builds allow you to customize the kernels in the build to provide smaller binary sizes - [see instructions](https://github.com/microsoft/onnxruntime/blob/master/docs/Reduced_Operator_Kernel_build.md).

### OpenMP (Deprecated)
#### Build Instructions
##### Windows

```powershell
.\build.bat --use_openmp
```

##### Linux/macOS

```bash
./build.sh --use_openmp
```

### DebugNodeInputsOutputs
OnnxRuntime supports build options for enabling debugging of intermediate tensor shapes and data.

#### Build Instructions
Set onnxruntime_DEBUG_NODE_INPUTS_OUTPUT to build with this enabled.

##### Linux

```bash
./build.sh --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
```

##### Windows

```
.\build.bat --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
```

#### Configuration

The debug dump behavior can be controlled with several environment variables.
See [onnxruntime/core/framework/debug_node_inputs_outputs_utils.h](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/framework/debug_node_inputs_outputs_utils.h) for details.

##### Examples

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

## Architectures
### 64-bit x86 

Also known as [x86_64](https://en.wikipedia.org/wiki/X86-64) or AMD64. This is the default.

### 32-bit x86
#### Build Instructions
##### Windows
* add `--x86` argument when launching `.\build.bat`

##### Linux
(Not officially supported)

---

### ARM

There are a few options for building for ARM.

* [Cross compiling for ARM with simulation (Linux/Windows)](#cross-compiling-for-arm-with-simulation-linuxwindows) - **Recommended**;  Easy, slow
* [Cross compiling on Linux](#cross-compiling-on-linux) - Difficult, fast
* [Native compiling on Linux ARM device](#native-compiling-on-linux-arm-device) - Easy, slower
* [Cross compiling on Windows](#cross-compiling-on-windows)

#### Cross compiling for ARM with simulation (Linux/Windows)

*EASY, SLOW, RECOMMENDED*

This method rely on qemu user mode emulation. It allows you to compile using a desktop or cloud VM through instruction level simulation. You'll run the build on x86 CPU and translate every ARM instruction to x86. This is much faster than compiling natively on a low-end ARM device and avoids out-of-memory issues that may be encountered. The resulting ONNX Runtime Python wheel (.whl) file is then deployed to an ARM device where it can be invoked in Python 3 scripts.

Here is [an example for Raspberrypi3 and Raspbian](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles/README.md#arm-32v7). Note: this does not work for Raspberrypi 1 or Zero, and if your operating system is different from what the dockerfile uses, it also may not work.

The build process can take hours.

#### Cross compiling on Linux

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

#### Native compiling on Linux ARM device

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

#### Cross compiling on Windows

**Using Visual C++ compilers**

1. Download and install Visual C++ compilers and libraries for ARM(64).
   If you have Visual Studio installed, please use the Visual Studio Installer (look under the section `Individual components` after choosing to `modify` Visual Studio) to download and install the corresponding ARM(64) compilers and libraries.

2. Use `.\build.bat` and specify `--arm` or `--arm64` as the build option to start building. Preferably use `Developer Command Prompt for VS` or make sure all the installed cross-compilers are findable from the command prompt being used to build using the PATH environmant variable.

---

### Android/iOS
Please see [Build for Android/iOS](./android-ios.md)
