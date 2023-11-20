---
title: Build for inferencing
parent: Build ONNX Runtime
nav_order: 1
redirect_from: /docs/how-to/build/inferencing
---

# Build ONNX Runtime for inferencing
{: .no_toc }

Follow the instructions below to build ONNX Runtime to perform inference.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## CPU
Basic CPU build

### Prerequisites
{: .no_toc }

* Checkout the source tree:

   ```bash
   git clone --recursive https://github.com/Microsoft/onnxruntime.git
   cd onnxruntime
   ```

* Install [Python 3.x](http://python.org/).

* Install [cmake-3.27](https://cmake.org/download/) or higher.

  On Windows, please run
  ```bat
    python -m pip install cmake
    where cmake
  ```
  
  On Linux, please run
  ```bat
    python3 -m pip install cmake
    which cmake
  ```
  If the above commands failed, please manually get cmake from https://cmake.org/download/.
  
  After the installation, you can run
  ```
    cmake --version
  ```
  to verify if the installation was successful.
    

### Build Instructions
{: .no_toc }

#### Windows

Open Developer Command Prompt for Visual Studio version you are going to use. This will properly setup the environment including paths to your compiler, linker, utilities and header files.
```
.\build.bat --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
```
The default Windows CMake Generator is Visual Studio 2022. 
For Visual Studio 2019 add `--cmake_generator "Visual Studio 16 2019"`. 

We recommend using Visual Studio 2022.

If you want to build an ARM64 binary on a Windows ARM64 machine, you can use the same command above. Just be sure that your Visual Studio, CMake and Python are all ARM64 version.

If you want to cross-compile an ARM32 or ARM64 or ARM64EC binary on a Windows x86 machine, you need to add "--arm" or "--arm64" or "--arm64ec" to the build command above. 

When building on x86 Windows without  "--arm" or "--arm64" or "--arm64ec" args, the built binaries will be 64-bit if your python is 64-bit, or 32-bit if your python is 32-bit, 

#### Linux

```
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
```

#### macOS

By default, ONNX Runtime is configured to be built for a minimum target macOS version of 10.12.
The shared library in the release Nuget(s) and the Python wheel may be installed on macOS versions of 10.12+.

If you would like to use [Xcode](https://developer.apple.com/xcode/) to build the onnxruntime for x86_64 macOS, please add the `--use_xcode` argument in the command line.

Without this flag, the cmake build generator will be Unix makefile by default.

Today, Mac computers are either Intel-Based or Apple silicon(aka. ARM) based. By default, ONNX Runtime's build script only generate bits for the CPU ARCH that the build machine has. If you want to do cross-compiling: generate ARM binaries on a Intel-Based Mac computer, or generate x86 binaries on a Mac ARM computer, you can set the "CMAKE_OSX_ARCHITECTURES" cmake variable. For example:

Build for Intel CPUs:
```bash
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=x86_64
```

Build for Apple silicon CPUs:
```bash
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64
```
Build for both:
```bash
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES="x86_64;arm64"
```
The last command will generate a fat-binary for both CPU architectures.

Note: unit tests will be skipped due to the incompatible CPU instruction set when doing cross-compiling.

#### Notes

* Please note that these instructions build the debug build, which may have performance tradeoffs. The "--config" parameter has four valid values: Debug, Release, RelWithDebInfo and MinSizeRel. Compared to "Release", "RelWithDebInfo" not only has debug info, it also disables some inlines to make the binary easier to debug. Thus RelWithDebInfo is slower than Release.
* To build the version from each release (which include Windows, Linux, and Mac variants), see these [.yml files](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/azure-pipelines/) for reference
* The build script runs all unit tests by default for native builds and skips tests by default for cross-compiled builds.
  To skip the tests, run with `--build` or `--update --build`.
* If you need to install protobuf from source code, please note:
   * First, please open [cmake/deps.txt](https://github.com/microsoft/onnxruntime/blob/main/cmake/deps.txt) to check which protobuf version ONNX Runtime's offical packages use. 
   * As we statically link to protobuf, on Windows protobuf's CMake flag `protobuf_BUILD_SHARED_LIBS` should be turned OFF, on Linux if the option is OFF you also need to make sure [PIC](https://cmake.org/cmake/help/latest/prop_tgt/POSITION_INDEPENDENT_CODE.html) is enabled. After the installation, you should have the 'protoc' executable in your PATH. It is recommended to run `ldconfig` to make sure protobuf libraries are found.
   * If you installed your protobuf in a non standard location it would be helpful to set the following env var:`export CMAKE_ARGS="-DONNX_CUSTOM_PROTOC_EXECUTABLE=full path to protoc"` so the ONNX build can find it. Also run `ldconfig <protobuf lib folder path>` so the linker can find protobuf libraries.
* If you'd like to install onnx from source code, install protobuf first and:
    ```
    export ONNX_ML=1
    python3 setup.py bdist_wheel
    pip3 install --upgrade dist/*.whl
    ```
   Then, it's better to uninstall protobuf before you start to build ONNX Runtime, especially if you have install a different version of protobuf other than what ONNX Runtime has.
---

## Supported architectures and build environments

### Architectures
{: .no_toc }

|           | x86_32       | x86_64       | ARM32v7      | ARM64        | PPC64LE |
|-----------|:------------:|:------------:|:------------:|:------------:|:-------:|
|Windows    | YES          | YES          |  YES         | YES          | NO      |
|Linux      | YES          | YES          |  YES         | YES          | YES     |
|macOS      | NO           | YES          |  NO          | NO           | NO      |
|Android      | NO           | NO          |  YES          | YES           | NO      |
|iOS      | NO           | NO          |  NO          | YES           | NO      |

### Build Environments(Host)
{: .no_toc }

| OS          | Supports CPU | Supports GPU| Notes                              |
|-------------|:------------:|:------------:|------------------------------------|
|Windows 10   | YES          | YES         | VS2019 through the latest VS2015 are supported |
|Windows 10 <br/> Subsystem for Linux | YES         | NO        |         |
|Ubuntu 20.x/22.x  | YES          | YES         | Also supported on ARM32v7 (experimental) |
|CentOS 7/8/9  | YES          | YES         | Also supported on ARM32v7 (experimental) |
|macOS        | YES          | NO         |    |

GCC 8.x and below are not supported.    
If you want to build a binary for a 32-bit architecture, you might have to do cross-compiling since a 32-bit compiler might not have enough memory to run the build.    
Building the code on Android/iOS is not supported. You need to use a Windows, Linux or macOS device to do so.


| OS/Compiler | Supports VC  | Supports GCC     |  Supports Clang  |
|-------------|:------------:|:----------------:|:----------------:|
|Windows 10   | YES          | Not tested       | Not tested       |
|Linux        | NO           | YES(gcc>=8)      | Not tested       |
|macOS        | NO           | Not tested       | YES (Minimum version required not ascertained)|

### Target Environments
You can build the code for
 - Windows
 - Linux
 - MacOS
 - Android
 - iOS
 - WebAssembly

At runtime:
- The minimum supported Windows version is Windows 10.
- The minimum supported CentOS version is 7.
- The minimum supported Ubuntu version is 16.04.

---

## Common Build Instructions

|Description|Command|Additional details|
|-----------|-----------|-----------|
|**Basic build**|`build.bat` (Windows)<br>`./build.sh` (Linux)||
|**Release build**|`--config` Release|Release build. Other valid config values are RelWithDebInfo and Debug.|
|**Build using parallel processing**|`--parallel`|This is strongly recommended to speed up the build.|
|**Build Shared Library**|`--build_shared_lib`||
|**Enable Training support**|`--enable_training`||

### APIs and Language Bindings

|API|Command|Additional details|
|-----------|-----------|-----------|
|**Python**|`--build_wheel`||
|**C# and C Nuget packages**|`--build_nuget`|Builds C# bindings and creates nuget package. Implies `--build_shared_lib` <br> Detailed instructions can be found [below](#build-nuget-packages).|
|**WindowsML**|`--use_winml`<br>`--use_dml`<br>`--build_shared_lib`|WindowsML depends on DirectML and the OnnxRuntime shared library|
|**Java**|`--build_java`|Creates an onnxruntime4j.jar in the build directory, implies `--build_shared_lib`<br>Compiling the Java API requires [gradle](https://gradle.org) v6.1+ to be installed in addition to the usual requirements.|
|**Node.js**|`--build_nodejs`|Build Node.js binding. Implies `--build_shared_lib`|


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

[Reduced Operator Kernel](./custom.md#reduce-operator-kernels) builds allow you to customize the kernels in the build to provide smaller binary sizes.


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
See [onnxruntime/core/framework/debug_node_inputs_outputs_utils.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/debug_node_inputs_outputs_utils.h) for details.

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

### ARM

There are a few options for building ONNX Runtime for ARM. 

First, you may do it on a real ARM device, or on a x86_64 device with an emulator(like qemu), or on a x86_64 device with a docker container with an emulator(you can run an ARM container on a x86_64 PC). Then the build instructions are essentially the same as the instructions for Linux x86_64. However, it wouldn't work if your the CPU you are targeting is not 64-bit since the build process needs more than 2GB memory.  

* [Cross compiling for ARM with simulation (Linux/Windows)](#cross-compiling-for-arm-with-simulation-linuxwindows) - **Recommended**;  Easy, slow, ARM64 only(no support for ARM32)
* [Cross compiling on Linux](#cross-compiling-on-linux) - Difficult, fast
* [Cross compiling on Windows](#cross-compiling-on-windows)

#### Cross compiling for ARM with simulation (Linux/Windows)

*EASY, SLOW, RECOMMENDED*

This method relies on qemu user mode emulation. It allows you to compile using a desktop or cloud VM through instruction level simulation. You'll run the build on x86 CPU and translate every ARM instruction to x86. This is much faster than compiling natively on a low-end ARM device. The resulting ONNX Runtime Python wheel (.whl) file is then deployed to an ARM device where it can be invoked in Python 3 scripts. The build process can take hours, and may run of memory if the target CPU is 32-bit.

#### Cross compiling on Linux

*Difficult, fast*

This option is very fast and allows the package to be built in minutes, but is challenging to setup. If you have a large code base (e.g. you are adding a new execution provider to onnxruntime), this may be the only feasible method.

1. Get the corresponding toolchain.

    TLDR; Go to https://www.linaro.org/downloads/, get "64-bit Armv8 Cortex-A, little-endian" and "Linux Targeted", not "Bare-Metal Targeted". Extract it to your build machine and add the bin folder to your $PATH env. Then skip this part.

    You can use [GCC](https://gcc.gnu.org/) or [Clang](http://clang.llvm.org/). Both work, but instructions here are based on GCC.

    In GCC terms:
    * "build" describes the type of system on which GCC is being configured and compiled.
    * "host" describes the type of system on which GCC runs.
    * "target" to describe the type of system for which GCC produce code.

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
   
2. (Optional) Setup sysroot to enable python extension. *Skip if not using Python.*

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

3. Generate CMake toolchain file
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

    Besides, you may also set CMAKE_SYSTEM_PROCESSOR. CMake's official document suggests when doing cross-compiling a CMAKE_TOOLCHAIN_FILE should set the CMAKE_SYSTEM_PROCESSOR variable to match target architecture that it specifies. But the document doesn't provide a list of valid values and we found this setting is not necessary. Anyway, if you know which value the variable should be set to, please add the setting there. ONNX Runtime's build scripts do not use this variable, but ONNX Runtime's dependencies may.

5.  Run CMake and make

    Append `-Donnxruntime_ENABLE_CPUINFO=OFF -DCMAKE_TOOLCHAIN_FILE=path/to/tool.cmake` to your cmake args, run cmake and make to build it. If you want to build Python package as well, you can use cmake args like:

    ```bash
    -Donnxruntime_GCC_STATIC_CPP_RUNTIME=ON -DCMAKE_BUILD_TYPE=Release -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_TOOLCHAIN_FILE=path/to/tool.cmake -Donnxruntime_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/mnt/pi/usr/bin/python3 -Donnxruntime_BUILD_SHARED_LIB=OFF -Donnxruntime_DEV_MODE=OFF "-DPYTHON_INCLUDE_DIR=/mnt/pi/usr/include;/mnt/pi/usr/include/python3.7m" -DNUMPY_INCLUDE_DIR=/mnt/pi/folder/to/numpy/headers
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


#### Cross compiling on Windows

**Using Visual C++ compilers**

1. Download and install Visual C++ compilers and libraries for ARM(64).
   If you have Visual Studio installed, please use the Visual Studio Installer (look under the section `Individual components` after choosing to `modify` Visual Studio) to download and install the corresponding ARM(64) compilers and libraries.

2. Use `.\build.bat` and specify `--arm` or `--arm64` as the build option to start building. Preferably use `Developer Command Prompt for VS` or make sure all the installed cross-compilers are findable from the command prompt being used to build using the PATH environmant variable.

---

### Mobile 
Please see [Build for Android](./android.md) and [Build for iOS](./ios.md)

### Web
Please see [Build for Web](./web.md)
