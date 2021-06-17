---
title: Custom Build
parent: Deploy ONNX Runtime Mobile
grand_parent: How to
has_children: false
nav_order: 4
---
{::options toc_levels="2..4" /}

# ONNX Runtime Mobile Custom Build

Creating a custom 'minimal' build of ONNX Runtime gives you control over what is included in order to minimize the binary size whilst satisfying the needs of your scenario. 

The configuration file that was generated during [model conversion](model-conversion) is used to specify the operators (and potentially the types) that the build will support.

The general ONNX Runtime inferencing [build instructions](../build/inferencing#build-instructions) apply, with additional options being specified to reduce the binary size.

## Contents
{: .no_toc}

* TOC
{:toc}


## Binary size reduction options

The follow options can be used to reduce the build size:

##### Enable the minimal build
  - `--minimal_build` [REQUIRED] 
    - A minimal build will ONLY support loading and executing ORT format models
    - RTTI is disabled by default in this build, unless the Python bindings (`--build_wheel`) are enabled.
    - If you wish to enable execution providers that compile kernels such as NNAPI or CoreML specify `--minimal_build extended`.
      - See [here](using-nnapi-coreml-with-ort-mobile) for details on using NNAPI and CoreML with ONNX Runtime Mobile

##### Reduce build to required operator kernels
  - `--include_ops_by_config` [REQUIRED] 
    - Add `--include_ops_by_config <config file produced during model conversion> --skip_tests` to the build parameters.
    - See the documentation on the [Reduced Operator Kernel build](../build/reduced) for more information on how this works. 
      - NOTE: Building will edit some of the ONNX Runtime source files to exclude unused kernels. If you wish to go back to creating a full build, or wish to change the operator kernels included, you MUST run `git reset --hard` or `git checkout HEAD -- ./onnxruntime/core/providers` from the root directory of your local ONNX Runtime repository to undo these changes.

##### Reduce types supported by the required operators
  - `--enable_reduced_operator_type_support` [OPTIONAL]
    - Enables [operator type reduction](model-conversion#enable-type-reduction).
        - NOTE: Requires ONNX Runtime version 1.7 or higher and for type reduction to have been enabled during model conversion

##### Disable exceptions
  - `--disable_exceptions` [OPTIONAL]
    - Disables support for exceptions in the build.
      - Any locations that would have thrown an exception will instead log the error message and call abort().
      - Requires `--minimal_build`.
      - NOTE: This is not a valid option if you need the Python bindings (`--build_wheel`) as the Python Wheel requires exceptions to be enabled.
    - Exceptions are only used in ONNX Runtime for exceptional things. If you have validated the input to be used, and validated that the model can be loaded, it is unlikely that ORT would throw an exception unless there's a system level issue (e.g. out of memory).

##### Disable ML operator support
  - `--disable_ml_ops` [OPTIONAL]
    - Whilst the operator kernel reduction script will disable all unused ML operator kernels, additional savings can be achieved by removing support for ML specific types. If you know that your model has no ML ops, or no ML ops that use the Map type, this flag can be provided.
    - See the specs for the [ONNX ML Operators](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md) if unsure.

##### Use shared libc++ on Android
  - `--android_cpp_shared` [OPTIONAL]
    - Building using the shared libc++ library instead of the default static libc++ library will result in a smaller libonnxruntime.so library.
    - See [Android NDK documentation](https://developer.android.com/ndk/guides/cpp-support) for more information.

## Build Configuration

The `MinSizeRel` configuration will produce the smallest binary size.<br>
The `Release` configuration can also be used if you wish to prioritize performance over binary size.

## Example build commands

### Windows

`<ONNX Runtime repository root>.\build.bat --config=MinSizeRel --cmake_generator="Visual Studio 16 2019" --build_shared_lib --minimal_build --disable_ml_ops --disable_exceptions --include_ops_by_config <config file from model conversion> --skip_tests`

### Linux

`<ONNX Runtime repository root>./build.sh --config=MinSizeRel --build_shared_lib --minimal_build --disable_ml_ops --disable_exceptions --include_ops_by_config <config file from model conversion> --skip_tests`

## Building ONNX Runtime Python Wheel

If you wish to use the ONNX Runtime python bindings with a minimal build, exceptions must be enabled due to Python requiring them.

Remove `--disable_exceptions` and add `--build_wheel` to the build command in order to build a Python Wheel with the ONNX Runtime bindings.

A .whl file will be produced in the build output directory under the `<config>/dist` folder.

  - The Python Wheel for a Windows MinSizeRel build using build.bat would be in `<ONNX Runtime repository root>\build\Windows\MinSizeRel\MinSizeRel\dist\`
  - The Python Wheel for a Linux MinSizeRel build using build.sh would be in `<ONNX Runtime repository root>/build/Linux/MinSizeRel/dist/`

The wheel can be installed using `pip`. Adjust the following command for your platform and the whl filename.
  -  `pip install -U .\build\Windows\MinSizeRel\MinSizeRel\dist\onnxruntime-1.7.0-cp37-cp37m-win_amd64.whl`

------

Next: [Model execution](model-execution)