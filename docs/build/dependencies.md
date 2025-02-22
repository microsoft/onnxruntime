# Dependency Management in ONNX Runtime

This document provides supplement information to CMake’s [“Using Dependencies Guide”](https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html) and it is more ONNX Runtime specific. 
Overall, there are three ways to get dependencies for an ONNX Runtime build:

1. [Use VCPKG](#vcpkg) (Recommended)
2. [Build everything from source](#build-everything-from-source)
3. [Use preinstalled packages](#use-preinstalled-packages) (For Advanced Users)

## VCPKG

### What is VCPKG?

VCPKG is a free and open-source C/C++ package manager maintained by Microsoft and the C++ community. It was mainly developed by the Visual Studio team at Microsoft. It helps developers manage their C++ dependencies in a simple and declarative way. It is based on CMake and can be integrated into your CMake project or used separately before building. ONNX Runtime uses the former approach, known as manifest mode.
VCPKG provides better support for cross-compiling. For example, if you are building ONNX Runtime for ARM64 on a x64 machine, VCPKG will compile protoc in x64, simplifying the build process.

### Prerequisites for using VCPKG

Refer to the VCPKG documentation for supported hosts: https://github.com/microsoft/vcpkg-docs/blob/main/vcpkg/concepts/supported-hosts.md. For example, on Ubuntu, you need to install the following packages:
apt-get install git curl zip unzip pkgconfig ninja-build

### How to build ONNX Runtime with VCPKG

Just add “--use_vcpkg” to your build command.  The build script(build.py) will check out a fresh vcpkg repo into your build directory and bootstrap the vcpkg tool.  If you encounter any errors, you may need to manually get VCPKG using the following steps:
1. Install Git and Run “git clone https://github.com/microsoft/vcpkg.git”
2. Navigate to the VCPKG directory and run the bootstrap script:
     - On Windows: bootstrap-vcpkg.bat
     - On other systems: bootstrap-vcpkg.sh

    If the script cannot find some prerequisites, install the missing software and try again.
3. Set the environment variable VCPKG_INSTALLATION_ROOT to the VCPKG directory, then go back to the ONNX Runtime source folder and run the build script again.
For more details, see: https://github.com/microsoft/vcpkg-docs/blob/main/vcpkg/get_started/includes/setup-vcpkg.md. If you get blocked on bootstrapping VCPKG, please contact the VCPKG team for support.

### VCPKG ports, triplets and toolchains

A package in VCPKG is called a VCPKG port. The build scripts for a port can be find at http://github.com/microsoft/vcpkg/tree/master/ports . ONNX Runtime also has some custom ports that are hosted at https://github.com/microsoft/onnxruntime/tree/main/cmake/vcpkg-ports. The custom ports have higher priority than the official ports. The files in a port's directory contains configurations that is specific to that port. For example, whether enable CUDA or not.

A triplet is a cmake file that contains configurations that are applied to all ports in the current build. For example, whether enable C++ exception or not. It is only used for building dependencies. It won't affect the build flags of ONNX Runtime's source code. The compiler flags and cmake variables set in tools/ci_build/build.py are for ONNX Runtime only and are not applicable to vcpkg ports. Therefore we need to use custom triplet files to make the settings consistent. 

A toolchain file is for setting up compilers/linkers etc, which is more powerful.  ONNX Runtime usually use standard vcpkg toolchain files, except for WebAssembly build.

### Limitations 

Currently the support the vcpkg is still development in progress. It does not support the following scenarios:

1.  When --minimal_build is enabled
2.  When --ios is enabled
3.  Windows WebGPU native build

And some dependencies are not managed by VCPKG yet. For example, Dawn.    

### Do I have to convert all dependencies to vcpkg ports?

If the dependency is used by an ONNX Runtime release package, then definitely yes. Otherwise we can discuss it case by case.

### Process for updating a VCPKG port

First, please check if the port is a custom port in the `cmake/vcpkg/vcpkg-ports` directory.  If yes, you need to update it there. And At least you need to update two places: the version number in the vcpkg.json file, and the SHA512 hash value in ports.cmake file. If you don't know which SHA512 value to put there, you could slightly modify the current hash value by flipping a few bytes, then build ONNX Runtime with the "--use_vcpkg" flag(do not use additional flags to enable any asset cache), then vcpkg will generate an error message to tell you the actual hash value that it expects.  You may also need to update the patch files. You can clone the library's repo, then checkout the new version that you are updating to, then apply the old patch files, then resolve the conflicts, the use "git diff" to generate a new patch file to overrite the existing one. 

If the dependency was from VCPKG's official registry(in https://github.com/microsoft/vcpkg), that would be much easier. Just open vcpkg-configuration.json and update the baseline commit id to the latest vcpkg commit id.

Then create a PR.  Then someone from the ONNX Runtime dev team will review your changes, replicate the dependencies to an internal place and trigger the pull request pipelines. If everything goes well, we will merge your change.  

## Build everything from source

Add “--cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER” to your build command. When VCPKG is not enabled, we use CMake’s FetchContent to manage dependencies. All such dependencies are listed in cmake/deps.txt, allowing you to customize versions and download URLs. This is useful for meeting network isolation requirements or upgrading/downgrading library versions.
When declaring a dependency in ONNX Runtime’s CMake files, if FIND_PACKAGE arguments are provided, FetchContent will use CMake’s FindPackage module to find dependencies from system locations. Add --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER to disable this behavior.

## Use preinstalled packages

This is the default mode when building ONNX Runtime from source. If a build has neither '--use_vcpkg' nor '--cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER', it will be in this mode. It is because FetchContent is a wrapper around various CMake Dependency Providers. By default it prefers to use [find_package](https://cmake.org/cmake/help/latest/command/find_package.html) if FIND_PACKAGE arguments are provided for the dependency.   If you are integrating ONNX Runtime into a package manager(like dnf), you will need to use this approach. However, it has some caveats:

1. ONNX Runtime has local patches for dependencies that will not be applied to your preinstalled libraries. Most patches are not necessary for basic functionality.
2. If you have installed a library version different from what ONNX Runtime expects, the build script cannot warn you. This may lead to strange build failures.
3. Each library can be built in different ways. For example, ONNX Runtime expects ONNX is built with “-DONNX_DISABLE_STATIC_REGISTRATION=ON”. If you have got a prebuilt ONNX library from somewhere, most likely it was not built in this way. 

Therefore we say it is for advanced users.
