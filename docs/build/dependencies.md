Dependency Management in ONNX Runtime
This document provides supplement information to CMake’s [“Using Dependencies Guide”](https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html) and it is more ONNX Runtime specific. 
Overall, there are three ways to get dependencies for an ONNX Runtime build:

1.	Use VCPKG (Recommended)
2.	Build everything from source
3.	Use preinstalled packages (Advanced Users Only)

This document will have one section for each above.

# VCPKG

## What is VCPKG?

VCPKG is a free and open-source C/C++ package manager maintained by Microsoft and the C++ community. It was mainly developed by the Visual Studio team at Microsoft. It helps developers manage their C++ dependencies in a simple and declarative way. It is based on CMake and can be integrated into your CMake project or used separately before building. ONNX Runtime uses the former approach, known as manifest mode.
VCPKG provides better support for cross-compiling. For example, if you are building ONNX Runtime for ARM64 on a x64 machine, VCPKG will compile protoc in x64, simplifying the build process.

## Prerequisites for using VCPKG

Refer to the VCPKG documentation for supported hosts: https://github.com/microsoft/vcpkg-docs/blob/main/vcpkg/concepts/supported-hosts.md. For example, on Ubuntu, you need to install the following packages:
apt-get install git curl zip unzip pkgconfig ninja-build

## How to build ONNX Runtime with VCPKG

Just add “--use_vcpkg” to your build command.  The build script(build.py) will check out a fresh vcpkg repo into your build directory and bootstrap the vcpkg tool.  If you encounter any errors, you may need to manually get VCPKG using the following steps:
1.	Install Git and Run “git clone https://github.com/microsoft/vcpkg.git”
2.	Navigate to the VCPKG directory and run the bootstrap script:
     - On Windows: bootstrap-vcpkg.bat
     - On other systems: bootstrap-vcpkg.sh

    If the script cannot find some prerequisites, install the missing software and try again.
3.	Set the environment variable VCPKG_INSTALLATION_ROOT to the VCPKG directory, then go back to the ONNX Runtime source folder and run the build script again.
For more details, see: https://github.com/microsoft/vcpkg-docs/blob/main/vcpkg/get_started/includes/setup-vcpkg.md. If you get blocked on bootstrapping VCPKG, please contact the VCPKG team for support.

# Build everything from source

Add “--cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER” to your build command. When VCPKG is not enabled, we use CMake’s FetchContent to manage dependencies. All such dependencies are listed in cmake/deps.txt, allowing you to customize versions and download URLs. This is useful for meeting network isolation requirements or upgrading/downgrading library versions.
When declaring a dependency in ONNX Runtime’s CMake files, if FIND_PACKAGE arguments are provided, FetchContent will use CMake’s FindPackage module to find dependencies from system locations. Add --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER to disable this behavior.

# Use preinstalled packages
FetchContent is a wrapper around various CMake Dependency Providers. By default, it prefers to use FindPackage. However, there are some caveats:
1.	ONNX Runtime has local patches for dependencies that will not be applied to your preinstalled libraries. Most patches are not necessary for basic functionality.
2.	If you have installed a library version different from what ONNX Runtime expects, the build script cannot warn you. This may lead to strange build failures.
3.	Each library can be built in different ways. For example, ONNX Runtime expects ONNX is built with “-DONNX_DISABLE_STATIC_REGISTRATION=ON”. If you have got a prebuilt ONNX library from somewhere, most likely it was not built in this way. 
