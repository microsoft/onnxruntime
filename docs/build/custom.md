---
title: Custom build
description: Customize the ONNX Runtime binaries, including building with a reduced set of operators
parent: Build ORT
nav_order: 7
redirect_from: /docs/tutorials/mobile/custom-build,/docs/build/reduced
---

# Build a custom ONNX Runtime package

The ONNX Runtime package can be customized when the demands of the target environment require it.

The most common scenario for customizing the ONNX Runtime build is for smaller footprint deployments, such as mobile and web.

And the most common mechanism to customize the build is to reduce the set of supported operators in the runtime to only those in the model or models that run in the target environment.

To build a custom ONNX Runtime package, the [build](./index.md) instructions apply, with some extra build options that are specified below.

## Reduced operator set

To reduce the compiled binary size of ONNX Runtime, the operator kernels included in the build can be reduced to just those required by your model/s.

The operators that are included are specified at build time, in a [configuration file](../reference/reduced-operator-config.md) that can be generated from a model or set of models.

### Build option to reduce build to required operator kernels

**`--include_ops_by_config` [REQUIRED]**

* Add `--include_ops_by_config <config file produced during model conversion> --skip_tests` to the build parameters.
* See the documentation on the [Reduced Operator Kernel build](../../build/reduced.md) for more information on how this works.
* NOTE: Building will edit some of the ONNX Runtime source files to exclude unused kernels. If you wish to go back to creating a full build, or wish to change the operator kernels included, you MUST run `git reset --hard` or `git checkout HEAD -- ./onnxruntime/core/providers` from the root directory of your local ONNX Runtime repository to undo these changes.

### Option to reduce types supported by the required operators

** `--enable_reduced_operator_type_support` [OPTIONAL]**

* Enables [operator type reduction](./model-conversion.md#enable-type-reduction). Requires ONNX Runtime version 1.7 or higher and for type reduction to have been enabled during model conversion

If the configuration file is created using ORT format models, the input/output types that individual operators require can be tracked if `--enable_type_reduction` is specified. This can be used to further reduce the build size if `--enable_reduced_operator_type_support` is specified when building ORT.

ONNX format models are not guaranteed to include the required per-node type information, so cannot be used with this option.

## Minimal build

ONNX Runtime can be built to further minimize the binary size, by only including support for loading and executing models in [ORT format](../reference/ort-format-model-conversion.md), and not ONNX format.

**`--minimal_build` [REQUIRED]**

* RTTI is disabled by default in this build, unless the Python bindings (`--build_wheel`) are enabled.
* If you wish to enable execution providers that compile kernels such as NNAPI or CoreML specify `--minimal_build extended`. See [here](./using-platform-specific-ep.html#using-nnapi-and-coreml-with-onnx-runtime-mobile) for details on using NNAPI and CoreML with ONNX Runtime Mobile

## Other customizations

### Disable exceptions
  
**`--disable_exceptions` [OPTIONAL]**

* Any locations that would have thrown an exception will instead log the error message and call abort().
* Requires `--minimal_build`.
* NOTE: This is not a valid option if you need the Python bindings (`--build_wheel`) as the Python Wheel requires exceptions to be enabled.
* Exceptions are only used in ONNX Runtime for exceptional things. If you have validated the input to be used, and validated that the model can be loaded, it is unlikely that ORT would throw an exception unless there's a system level issue (e.g. out of memory).

### Disable ML operator support

**`--disable_ml_ops` [OPTIONAL]**

* Whilst the operator kernel reduction script disables all unused ML operator kernels, additional savings can be achieved by removing support for ML specific types. If you know that your model has no ML ops, or no ML ops that use the Map type, this flag can be provided.
* See the specs for the [ONNX ML Operators](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md) if unsure.

### Use shared libc++ on Android

**`--android_cpp_shared` [OPTIONAL]**

* Building using the shared libc++ library instead of the default static libc++ library results in a smaller libonnxruntime.so library.
* See [Android NDK documentation](https://developer.android.com/ndk/guides/cpp-support) for more information.

## Build Configuration

**--config** option

The `MinSizeRel` configuration will produce the smallest binary size.

The `Release` configuration can also be used if you wish to prioritize performance over binary size.

## Custom build package format

### Web

TODO

### iOS

TODO

### Android

TODO

### Python

If you wish to use the ONNX Runtime python bindings with a minimal build, exceptions must be enabled due to Python requiring them.

Remove `--disable_exceptions` and add `--build_wheel` to the build command in order to build a Python Wheel with the ONNX Runtime bindings.

A .whl file will be produced in the build output directory under the `<config>/dist` folder.

* The Python Wheel for a Windows MinSizeRel build using build.bat would be in `<ONNX Runtime repository root>\build\Windows\MinSizeRel\MinSizeRel\dist\`
* The Python Wheel for a Linux MinSizeRel build using build.sh would be in `<ONNX Runtime repository root>/build/Linux/MinSizeRel/dist/`

The wheel can be installed using `pip`. Adjust the following command for your platform and the whl filename.

```bash
pip install -U .\build\Windows\MinSizeRel\MinSizeRel\dist\onnxruntime-1.7.0-cp37-cp37m-win_amd64.whl
```

## Version of ONNX Runtime to build from

Unless there is a specific feature you need, do not use the unreleased 'master' branch.

Once you have cloned the ONNX Runtime repo, checkout one of the release branches to build from.

```bash
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
git checkout <release branch>
```

where `release branch` is one of the values in the `Branch` column:

| Release | Date | Branch |
|---------|------|--------|
| 1.9 | 2021-09-22 | rel-1.9.1 |
| 1.8 | 2021-06-02 | rel-1.8.2 |
| 1.7 | 2021-03-03 | rel-1.7.2 |
| 1.6 | 2020-12-11 | rel-1.6.0 |
| 1.5 | 2020-10-30 | rel-1.5.3 |

## Example build commands

### Build on windows, with reduced operator support, and support for ORT format models only

`<ONNX Runtime repository root>.\build.bat --config=MinSizeRel --cmake_generator="Visual Studio 16 2019" --build_shared_lib --minimal_build --disable_ml_ops --disable_exceptions --include_ops_by_config <config file from model conversion> --skip_tests`

### Linux

`<ONNX Runtime repository root>./build.sh --config=MinSizeRel --build_shared_lib --minimal_build --disable_ml_ops --disable_exceptions --include_ops_by_config <config file from model conversion> --skip_tests`

