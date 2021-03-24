This directory contains a few C/C++ sample applications for demoing onnxruntime usage:

1. (Windows only) fns_candy_style_transfer: A C application that uses the FNS-Candy style transfer model to re-style images. 
2. (Windows only) MNIST: A windows GUI application for doing handwriting recognition
3. (Windows only) imagenet: An end-to-end sample for the [ImageNet Large Scale Visual Recognition Challenge 2012](http://www.image-net.org/challenges/LSVRC/2012/) - requires ATL libraries to be installed as a part of the VS Studio installation.
4. model-explorer: A commandline C++ application that generates random data and performs model inference. A second C++ application demonstrates how to perform batch processing.

# How to build

## Prerequisites
1. Visual Studio 2015/2017/2019
2. cmake(version >=3.13)
3. (optional) [libpng 1.6](http://www.libpng.org/pub/png/libpng.html)

You may get a precompiled libpng library from [https://onnxruntimetestdata.blob.core.windows.net/models/libpng.zip](https://onnxruntimetestdata.blob.core.windows.net/models/libpng.zip)

## Install ONNX Runtime
You may either get a prebuit onnxruntime from nuget.org, or build it from source by following the [build instructions](https://www.onnxruntime.ai/docs/how-to/build.html).
If you build it by yourself, you must append the "--build_shared_lib" flag to your build command. 
Open Developer Command Prompt for Visual Studio version you are going to use. This will setup necessary environment for the compiler and other things to be found.
```
build.bat --config RelWithDebInfo --build_shared_lib --parallel 
```

By default this will build a project with "C:\Program Files (x86)\onnxruntime" install destination. This is a protected folder on Windows. If you do not want to run installation with elevated priviliges you will need to override the default installation location by passing extra CMake arguments. For example:

```
build.bat --config RelWithDebInfo --build_shared_lib --parallel  --cmake_extra_defines CMAKE_INSTALL_PREFIX=c:\dev\ort_install
```

By default products of the build on Windows go to .\build\Windows\<config> folder. In the case above it would be .\build\Windows\RelWithDebInfo.
If you did not specify alternative installation location above you would need to open an elevated command prompt to install onnxruntime.
Run the following commands.

```
cd .\Windows\RelWithDebInfo
msbuild INSTALL.vcxproj /p:Configuration=RelWithDebInfo
```

## Build the samples

Open Developer Command Prompt for Visual Studio version you are going to use, change your current directory to samples\c_cxx, then run
```bat
mkdir build && cd build
cmake .. -A x64 -T host=x64 -DLIBPNG_ROOTDIR=C:\path\to\your\libpng\binary -DONNXRUNTIME_ROOTDIR=c:\dev\ort_install
```
You may omit the "-DLIBPNG_ROOTDIR=..." argument if you don't have the libpng library.
You may omit "-DONNXRUNTIME_ROOTDIR=..." if you installed to a default location.

You may append "-Donnxruntime_USE_CUDA=ON" or "-Donnxruntime_USE_DML=ON" to the last command args if your onnxruntime binary was built with CUDA or DirectML support respectively.

You can then either open the solution in a Visual Studio and build it from there
```bat
devenv onnxruntime_samples.sln
```
Or build it using msbuild

```bat
msbuild onnxruntime_samples.sln /p:Configuration=Debug|Release
```

To run the samples make sure that your Install Folder Bin is in the path so your sample executable can find onnxruntime dll and libpng if you used it.

