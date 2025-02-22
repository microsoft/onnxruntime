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

* Install [Python 3.10+](https://python.org/).

* Install [cmake-3.28](https://cmake.org/download/) or higher. 

  On Windows, we recommend getting the latest version from WinGet. Please run
  ```bat
    winget install -e --id Kitware.CMake
  ```
  
  On Linux, you may get it from pypi. Please run
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
The default Windows CMake Generator is Visual Studio 2022.  The other Visual Studio versions are not supported.

If you want to build an ARM64 binary on a Windows ARM64 machine, you can use the same command above. Just be sure that your Visual Studio, CMake and Python are all ARM64 version.

If you want to cross-compile an ARM64 or ARM64EC binary on a Windows x86 machine, you need to add "--arm64" or "--arm64ec" to the build command above. 

Please make sure your python interpreter is a 64-bit Windows application. We no longer support 32-bit build.

#### Linux

```
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
```

#### macOS

By default, ONNX Runtime is configured to be built for a minimum target macOS version of 10.12.
The shared library in the release Nuget(s) and the Python wheel may be installed on macOS versions of 10.12+.

If you would like to use [Xcode](https://developer.apple.com/xcode/) to build the onnxruntime for x86_64 macOS, please add the `--use_xcode` argument in the command line.

Without this flag, the cmake build generator will be Unix makefile by default.

Today, Mac computers are either Intel-Based or Apple silicon-based. By default, ONNX Runtime's build script only generate bits for the CPU ARCH that the build machine has. If you want to do cross-compiling: generate arm64 binaries on a Intel-Based Mac computer, or generate x86 binaries on a Mac
system with Apple silicon, you can set the "CMAKE_OSX_ARCHITECTURES" cmake variable. For example:

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

#### AIX
In AIX, you can build ONNX Runtime for 64bit using

* IBM Open XL compiler tool chain.
  Minimum required AIX OS version is 7.2. You need to have 17.1.2 compiler PTF5 (17.1.2.5) version.
* GNU GCC compiler tool chain.
  Minimum required AIX OS version is 7.3. GCC version 10.3+ is required.

For IBM Open XL, export below environment settings.
```bash
ulimit -m unlimited
ulimit -d unlimited
ulimit -n 2000
ulimit -f unlimited
export OBJECT_MODE=64
export BUILD_TYPE="Release"
export CC="/opt/IBM/openxlC/17.1.2/bin/ibm-clang" 
export CXX="/opt/IBM/openxlC/17.1.2/bin/ibm-clang++_r"
export CFLAGS="-pthread -m64 -D_ALL_SOURCE -mcmodel=large -Wno-deprecate-lax-vec-conv-all  -Wno-unused-but-set-variable -Wno-unused-command-line-argument -maltivec -mvsx  -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare"
export CXXFLAGS="-pthread -m64 -D_ALL_SOURCE -mcmodel=large -Wno-deprecate-lax-vec-conv-all -Wno-unused-but-set-variable -Wno-unused-command-line-argument -maltivec -mvsx  -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare"
export LDFLAGS="-L$PWD/build/Linux/$BUILD_TYPE/ -lpthread"
export LIBPATH="$PWD/build/Linux/$BUILD_TYPE/"
```
For GCC, export below environment settings.
```bash
ulimit -m unlimited
ulimit -d unlimited
ulimit -n 2000
ulimit -f unlimited
export OBJECT_MODE=64
export BUILD_TYPE="Release"
export CC="gcc" 
export CXX="g++"
export CFLAGS="-maix64 -pthread -DFLATBUFFERS_LOCALE_INDEPENDENT=0 -maltivec -mvsx   -Wno-unused-function -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare -fno-extern-tls-init -Wl,-berok "
export CXXFLAGS="-maix64 -pthread -DFLATBUFFERS_LOCALE_INDEPENDENT=0 -maltivec -mvsx  -Wno-unused-function -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare -fno-extern-tls-init -Wl,-berok "
export LDFLAGS="-L$PWD/build/Linux/$BUILD_TYPE/ -Wl,-bbigtoc -lpython3.9"
export LIBPATH="$PWD/build/Linux/$BUILD_TYPE"
```
To initiate build, run the below command
```bash
./build.sh \
--config $BUILD_TYPE\
  --build_shared_lib \
  --skip_submodule_sync \
  --cmake_extra_defines CMAKE_INSTALL_PREFIX=$PWD/install \
  --parallel  
```

* If you want to install the package in a custom directory, then mention the directory location as value of CMAKE_INSTALL_PREFIX.
* In case of IBM Open XL compiler tool chain, It is possible that in AIX 7.2 some of the runtime libraries like libunwind.a  needed for onnxruntime, will be missing. To fix this, you can install the relevant file-sets.
* --parallel option in build option.
  As name suggest, this option is for parallel building and resource intensive option. So, if your system is not having good amount of memory for each CPU core, then this option can be skipped. 
* --allow_running_as_root  is needed if root user is triggering the build.
    

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

|           | x86_32       | x86_64       | ARM32v7      | ARM64        | PPC64LE | RISCV64 | PPC64BE | S390X |
|-----------|:------------:|:------------:|:------------:|:------------:|:-------:|:-------:| :------:|:-----:|
|Windows    | YES          | YES          |  YES         | YES          | NO      | NO      | NO      | NO    |
|Linux      | YES          | YES          |  YES         | YES          | YES     | YES     | NO      | YES   |
|macOS      | NO           | YES          |  NO          | NO           | NO      | NO      | NO      | NO    |
|Android      | NO           | NO          |  YES          | YES           | NO      | NO      | NO     | NO    |
|iOS      | NO           | NO          |  NO          | YES           | NO      | NO      |  NO     | NO    |
|AIX        | NO           | NO          |  NO          | NO           | NO      | NO      |  YES     | NO    |

### Build Environments(Host)
{: .no_toc }

| OS          | Supports CPU | Supports GPU| Notes                              |
|-------------|:------------:|:------------:|------------------------------------|
|Windows 10   | YES          | YES         | VS2019 through the latest VS2022 are supported |
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

* You may build onnxruntime with VCPKG. The document [/docs/build/dependencies.md](/docs/build/dependencies.md) has more information about it.


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

### RISC-V

   Running on a platform like RISC-V involves cross-compiling from a host platform (e.g. Linux) to a target platform (a specific RISC-V CPU architecture and operating system). Follow the instructions below to build ONNX Runtime for RISC-V 64-bit.

#### Cross compiling on Linux

1. Download and install GNU RISC-V cross-compile toolchain and emulator.

    If you are using Ubuntu, you can obtain a prebuilt RISC-V GNU toolchain by visiting https://github.com/riscv-collab/riscv-gnu-toolchain/releases. Look for the appropriate version, such as `riscv64-glibc-ubuntu-22.04-llvm-nightly-XXXX-nightly.tar.gz`. After downloading, extract the archive to your build machine and add the "bin" folder to your $PATH environment variable. The QEMU emulator is also located in the toolchain folder. Skip this step if you've followed these instructions.

    For users with other Linux distributions, the option to compile GCC from its source code is available. It is recommended to choose the a compiler version that corresponds to your target operating system, and opting for the latest stable release is recommended.

    When you get the compiler, run `riscv64-unknown-linux-gnu-gcc -v` This should produce an output like below:
    ```bash
       Using built-in specs.
       COLLECT_GCC=/path/to/riscv64-unknown-linux-gnu-gcc
       COLLECT_LTO_WRAPPER=/path/to/libexec/gcc/riscv64-unknown-linux-gnu/13.2.0/lto-wrapper
       Target: riscv64-unknown-linux-gnu
       Configured with: /path/to/riscv-gnu-toolchain/gcc/configure --target=riscv64-unknown-linux-gnu --prefix=/opt/riscv --with-sysroot=/opt/riscv/sysroot --with-pkgversion= --with-system-zlib --enable-shared --enable-tls --enable-languages=c,c++,fortran --disable-libmudflap --disable-libssp --disable-libquadmath --disable-libsanitizer --disable-nls --disable-bootstrap --src=.././gcc --disable-multilib --with-abi=lp64d --with-arch=rv64gc --with-tune=rocket --with-isa-spec=20191213 'CFLAGS_FOR_TARGET=-O2    -mcmodel=medlow' 'CXXFLAGS_FOR_TARGET=-O2    -mcmodel=medlow'
       Thread model: posix
       Supported LTO compression algorithms: zlib zstd
       gcc version 13.2.0 ()
    ```

2. Configure the RISC-V 64-bit CMake toolchain file

    The CMake toolchain file is located at `${ORT_ROOT}/cmake/riscv64.toolchain.cmake`. If there are alternative values for the variable that need to be configured, please include the corresponding settings in that file.

3. Cross compiling on Linux

    Execute `./build.sh` with the `--rv64` flag, specify the RISC-V toolchain root using `--riscv_toolchain_root`, and provide the QEMU path using `--riscv_qemu_path` as build options to initiate the build process. Make sure that all installed cross-compilers are accessible from the command prompt used for building by configuring the `PATH` environment variable.

    (Optional) If you intend to utilize xnnpack as the execution provider, be sure to include the `--use_xnnpack` option in your build configuration.

    Example build command:
    ```bash
    ./build.sh --parallel --config Debug --rv64 --riscv_toolchain_root=/path/to/toolchain/root --riscv_qemu_path=/path/to/qemu-riscv64 --skip_tests
    ```


### Arm

There are a few options for building ONNX Runtime for Arm®-based devices. 

First, you may do it on a real Arm-based device, or on a x86_64 device with an emulator(like qemu), or on a x86_64 device with a docker container with an emulator(you can run an Arm-based container on a x86_64 PC). Then the build instructions are essentially the same as the instructions for Linux x86_64. However, it wouldn't work if your the CPU you are targeting is not 64-bit since the build process needs more than 2GB memory.  

* [Cross compiling for Arm-based devices with simulation (Linux/Windows)](#cross-compiling-for-arm-based-devices-with-simulation-linuxwindows) - **Recommended**;  Easy, slow, ARM64 only(no support for ARM32)
* [Cross compiling on Linux](#cross-compiling-on-linux) - Difficult, fast
* [Cross compiling on Windows](#cross-compiling-on-windows)

#### Cross compiling for Arm-based devices with simulation (Linux/Windows)

*EASY, SLOW, RECOMMENDED*

This method relies on qemu user mode emulation. It allows you to compile using a desktop or cloud VM through instruction level simulation. You'll run the build on x86 CPU and translate every Arm architecture instruction to x86. This is potentially much faster than compiling natively on a low-end device. The resulting ONNX Runtime Python wheel (.whl) file is then deployed to an Arm-based device where it can be invoked in Python 3 scripts. The build process can take hours, and may run of memory if the target CPU is 32-bit.

#### Cross compiling on Linux

*Difficult, fast*

This option is very fast and allows the package to be built in minutes, but is challenging to setup. If you have a large code base (e.g. you are adding a new execution provider to onnxruntime), this may be the only feasible method.

1. Get the corresponding toolchain.

    TLDR; Go to https://www.linaro.org/downloads/, get "64-bit Armv8 Cortex-A, little-endian" and "Linux Targeted", not "Bare-Metal Targeted". Extract it to your build machine and add the bin folder to your $PATH env. Then skip this part.

    You can use [GCC](https://gcc.gnu.org/) or [Clang](https://clang.llvm.org/). Both work, but instructions here are based on GCC.

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
    Configured with: ../gcc-9.2.1-20190827/configure --bindir=/usr/bin --build=x86_64-redhat-linux-gnu --datadir=/usr/share --disable-decimal-float --disable-dependency-tracking --disable-gold --disable-libgcj --disable-libgomp --disable-libmpx --disable-libquadmath --disable-libssp --disable-libunwind-exceptions --disable-shared --disable-silent-rules --disable-sjlj-exceptions --disable-threads --with-ld=/usr/bin/aarch64-linux-gnu-ld --enable-__cxa_atexit --enable-checking=release --enable-gnu-unique-object --enable-initfini-array --enable-languages=c,c++ --enable-linker-build-id --enable-lto --enable-nls --enable-obsolete --enable-plugin --enable-targets=all --exec-prefix=/usr --host=x86_64-redhat-linux-gnu --includedir=/usr/include --infodir=/usr/share/info --libexecdir=/usr/libexec --localstatedir=/var --mandir=/usr/share/man --prefix=/usr --program-prefix=aarch64-linux-gnu- --sbindir=/usr/sbin --sharedstatedir=/var/lib --sysconfdir=/etc --target=aarch64-linux-gnu --with-bugurl=https://bugzilla.redhat.com/bugzilla/ --with-gcc-major-version-only --with-isl --with-newlib --with-plugin-ld=/usr/bin/aarch64-linux-gnu-ld --with-sysroot=/usr/aarch64-linux-gnu/sys-root --with-system-libunwind --with-system-zlib --without-headers --enable-gnu-indirect-function --with-linker-hash-style=gnu
    Thread model: single
    gcc version 9.2.1 20190827 (Red Hat Cross 9.2.1-3) (GCC)
    ```

    Check the value of `--build`, `--host`, `--target`, and if it has special args like `--with-arch=armv8-a`, `--with-arch=armv6`, `--with-tune=arm1176jz-s`, `--with-fpu=vfp`, `--with-float=hard`.

    You must also know what kind of flags your target hardware need, which can differ greatly. For example, if you just get the normal ARMv7 compiler and use it for Raspberry Pi V1 directly, it won't work because Raspberry Pi only has ARMv6. Generally every hardware vendor will provide a toolchain; check how that one was built.

    A target env is identified by:

    * Arch: x86_32, x86_64, armv6,armv7,arvm7l,aarch64,...
    * OS: bare-metal or linux.
    * Libc: gnu libc/ulibc/musl/...
    * ABI: Arm has multiple ABIs like eabi, eabihf...

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

1. Download and install Visual C++ compilers and libraries for Arm(64).
   If you have Visual Studio installed, please use the Visual Studio Installer (look under the section `Individual components` after choosing to `modify` Visual Studio) to download and install the corresponding Arm(64) compilers and libraries.

2. Use `.\build.bat` and specify `--arm` or `--arm64` as the build option to start building. Preferably use `Developer Command Prompt for VS` or make sure all the installed cross-compilers are findable from the command prompt being used to build using the PATH environmant variable.

3. Add `--use_vcpkg` to your build command, which can avoid manually handling protoc.exe.

---

### Mobile 
Please see [Build for Android](./android.md) and [Build for iOS](./ios.md)

### Web
Please see [Build for Web](./web.md)
