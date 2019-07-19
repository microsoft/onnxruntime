This directory contains a few C/C++ sample applications for demoing onnxruntime usage:

1. fns_candy_style_transfer: A C application that uses the FNS-Candy style transfer model to re-style images. 
2. MNIST: A windows GUI application for doing handwriting recognition
3. imagenet: An end-to-end sample for the [ImageNet Large Scale Visual Recognition Challenge 2012](http://www.image-net.org/challenges/LSVRC/2012/)

# How to build


## Install ONNX Runtime
You may either get a prebuit onnxruntime from nuget.org, or build it from source by following the [BUILD.md document](../../../BUILD.md). 
If you build it by yourself, you must append the "--build_shared_lib" flag to your build command. Like:

```
build.bat --config RelWithDebInfo --build_shared_lib --parallel
```
When the build is done, run Visual Studio as administrator and open the onnxruntime.sln file in your build directory.
![vs.png](vs.png)

Then select the "INSTALL" project and build it.  It will install your onnxruntime to  "C:\Program Files (x86)\onnxruntime"

