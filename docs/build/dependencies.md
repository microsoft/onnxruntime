# Dependency Management in ONNX Runtime

This document provides supplement information to CMake’s [“Using Dependencies Guide”](https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html) and it is more ONNX Runtime specific. 
Overall, there are three ways to get dependencies for an ONNX Runtime build:

1. [Use VCPKG](#vcpkg) (Recommended)
2. [Build everything from source](#build-everything-from-source)
3. [Use preinstalled packages](#use-preinstalled-packages) (For Advanced Users)

Below is a quick comparison:

|                           | Support Network Isolation[^1] | Support Binary Cache[^2] | Support Cross-compiling    | Developement Status                         |
|---------------------------|-------------------|--------------|--------------------|---------------------------------------------|
| VCPKG                     | YES               | Yes          | Good               | In-progress                                 |
| Everything From Source    | Partial[^3]           | No           | Just works[^4]         | Fully supported                             |
| Use Preinstalled Packages | YES               | Yes          | Difficult to setup | Some packages cannot be handled by this way |

[^1]: Can ONNX Runtime be built in an isolated network environment(without accessing the public internet)?
[^2]: Can the dependency libraries be built only once if they remain unchanged?
[^3]: For example, ONNX Runtime's native WebGPU does not support building in an isolated network.  Because the EP depends on [Dawn](https://dawn.googlesource.com/dawn) which is difficult to handle. 
[^4]: It works today, but ONNX Runtime has many EPs and dependencies. Over time, maintaining the current status has become difficult.

If your software needs to meet the U.S. President's Executive Order (EO) 14028 on Improving the Nation's Cybersecurity, we highly recommend using VCPKG. 

## VCPKG

### What is VCPKG?

VCPKG is a free and open-source C/C++ package manager maintained by Microsoft and the C++ community. It was mainly developed by the Visual Studio team at Microsoft. It helps developers manage their C++ dependencies in a simple and declarative way. It is based on CMake and can be integrated into your CMake project or used separately before building. ONNX Runtime uses the former approach, known as manifest mode.

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

### Unique Features
Compared to the other solutions listed in this page, VCPKG provides some unique features that we really want to have:

VCPKG provides better support for cross-compiling. For example, ONNX Runtime depends ONNX.  ONNX's source code has a few *.proto files. When building ONNX from source, we need to use protoc to generate C++ source files from the *.proto files. So, we need to build protoc and protoc's dependencies for the host OS. For example, if we are building an arm64 package on an x64 machine, we need to build protoc for x64 instead of arm64. And because protoc depends on libprotobuf, we must build libprotobuf twice for each CPU architecture. Whether using vcpkg or not, it must be built twice. CMake doesn't handle such scenarios, which added extra complexity to our build system. Now we can use vcpkg to solve this problem. It directly works fine out-of-box. 

With VCPKG, we only need to declare the root dependencies. Before moving to VCPKG, we needed to add all transitive dependencies to cmake/deps.txt and the cmake files under cmake/external folder to meet network isolation requirements(so that we could have an easy way to find all the download URLs) and [component detection](https://github.com/microsoft/component-detection) requirements. Now it is no longer needed because VCPKG has builtin support for asset cache and [SBOM generation](https://learn.microsoft.com/en-us/vcpkg/reference/software-bill-of-materials). 

VCPKG enforces that one library can only have one version. For example, the protobuf library used by onnxruntime_provider_openvino.dll and onnxruntime.dll must be exactly the same. Though it is stricter than necessary, it helps prevent ODR violation problems. It provides more benefit than dealing with potential conflicts and inconsistencies that arise from using multiple versions of the same library.

### Limitations 

Currently the support the vcpkg is still development in progress. It does not support the following scenarios:

1.  Minimal build
2.  iOS build
3.  Windows WebGPU native build

And some dependencies are not managed by VCPKG yet. For example, Dawn.   

When building for webassembly, it assumes the "--enable_wasm_simd" flag and "--enable_wasm_threads" flag are always set. It supports much less build varients than the second mode([Build everything from source](#build-everything-from-source)) 

Additionally, in this mode, the python interpreter used for running tools/ci_build/build.py might differ from the one used for building the VCPKG ports. This inconsistency can cause issues. Therefore, if you have multiple Python installations, we recommend adding the desired version to the beginning of your `PATH` to set it as the default.

It doesn't support setting VC toolset version or Windows SDK version yet. 

The support for Windows ARM64EC(including ARM64X) is experimental and is not well tested yet.

While the standard cmake has 4 different build types(Debug, Release, RelWithDebInfo and MinSizeRel), vcpkg only supports two. Therefore you may see binary size getting increase when building ONNX Runtime for RelWithDebInfo or MinSizeRel. This issue can be addressed by doing more customizations in the custom triplet files.  

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
