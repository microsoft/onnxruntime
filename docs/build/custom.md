---
title: Custom build
description: Customize the ONNX Runtime binaries, including building with a reduced set of operators
parent: Build ONNX Runtime
nav_order: 7
redirect_from: /docs/tutorials/mobile/custom-build,/docs/build/reduced,,/docs/tutorials/mobile/limitations
---

# Build a custom ONNX Runtime package
{: .no_toc }

The ONNX Runtime package can be customized when the demands of the target environment require it.

The most common scenario for customizing the ONNX Runtime build is for smaller footprint deployments, such as mobile and web.

And the most common mechanism to customize the build is to reduce the set of supported operators in the runtime to only those in the model or models that run in the target environment.

To build a custom ONNX Runtime package, the [build from source](./index.md) instructions apply, with some extra options that are specified below.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Reduce operator kernels

To reduce the compiled binary size of ONNX Runtime, the operator kernels included in the build can be reduced to just those required by your model/s.

The operators that are included are specified at build time, in a [configuration file](../reference/operators/reduced-operator-config-file.md) that can be generated from a model or set of models.

### Build option to reduce build to required operator kernels

**`--include_ops_by_config`**

* Add `--include_ops_by_config <config file produced during model conversion> --skip_tests` to the build parameters.

* NOTE: Building will edit some of the ONNX Runtime source files to exclude unused kernels.

  In particular, this source modification will be done during the "update" build phase, which is enabled by default or explicitly with the `--update` build parameter.

  **ONNX Runtime version 1.10 and earlier:** The source files are modified directly. If you wish to go back to creating a full build, or wish to change the operator kernels included, you MUST run `git reset --hard` or `git checkout HEAD -- ./onnxruntime/core/providers` from the root directory of your local ONNX Runtime repository to undo these changes.

  **ONNX Runtime version 1.11 and later:** Updated versions of the source files are generated in the build directory so there is no need to undo source file changes.

### Option to reduce types supported by the required operators

**`--enable_reduced_operator_type_support`**

* Enables [operator type reduction](../performance/model-optimizations/ort-format-models.md#enable-type-reduction). Requires ONNX Runtime version 1.7 or higher and for type reduction to have been enabled during model conversion

If the configuration file is created using ORT format models, the input/output types that individual operators require can be tracked if `--enable_type_reduction` is specified. This can be used to further reduce the build size if `--enable_reduced_operator_type_support` is specified when building ORT.

ONNX format models are not guaranteed to include the required per-node type information, so cannot be used with this option.

## Minimal build

ONNX Runtime can be built to further minimize the binary size.
These reduced size builds are called minimal builds and there are different minimal build levels described below.

### Basic

**`--minimal_build`**

RTTI is disabled by default in this build, unless the Python bindings (`--build_wheel`) are enabled.

A basic minimal build has the following limitations:

* No support for ONNX format models. The model must be converted to [ORT format](../performance/model-optimizations/ort-format-models.md).
* No support for runtime optimizations. Optimizations are performed during conversion to ORT format.
* Support for execution providers that statically register kernels (e.g. ONNX Runtime CPU Execution Provider) only.

### Extended

**`--minimal_build extended`**

An extended minimal build supports more functionality than a basic minimal build:

* Limited support for runtime partitioning (assigning nodes in a model to specific execution providers).
* Additional support for execution providers that compile kernels such as [NNAPI](../execution-providers/NNAPI-ExecutionProvider.md) and [CoreML](../execution-providers/CoreML-ExecutionProvider.md).
* **ONNX Runtime version 1.11 and later**: Limited support for runtime optimizations, via saved runtime optimizations and a few graph optimizers that are enabled at runtime.

## Other customizations

### Disable exceptions

**`--disable_exceptions`**

* Any locations that would have thrown an exception will instead log the error message and call abort().
* Requires `--minimal_build`.
* NOTE: This is not a valid option if you need the Python bindings (`--build_wheel`) as the Python Wheel requires exceptions to be enabled.
* Exceptions are only used in ONNX Runtime for exceptional things. If you have validated the input to be used, and validated that the model can be loaded, it is unlikely that ORT would throw an exception unless there's a system level issue (e.g. out of memory).

### Disable ML operator support

**`--disable_ml_ops`**

* Whilst the operator kernel reduction script disables all unused ML operator kernels, additional savings can be achieved by removing support for ML specific types. If you know that your model has no ML ops, or no ML ops that use the Map type, this flag can be provided.
* See the specs for the [ONNX ML Operators](https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md) if unsure.

### Use shared libc++ on Android

**`--android_cpp_shared`**

* Building using the shared libc++ library instead of the default static libc++ library results in a smaller libonnxruntime.so library.
* See [Android NDK documentation](https://developer.android.com/ndk/guides/cpp-support) for more information.

## Build Configuration

**`--config`**

The `MinSizeRel` configuration will produce the smallest binary size.

The `Release` configuration can also be used if you wish to prioritize performance over binary size.

## Version of ONNX Runtime to build from

Unless there is a specific feature you need, do not use the unreleased `main` branch.

Once you have cloned the ONNX Runtime repo, check out one of the release tags to build from.

```bash
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
git checkout <release tag>
```

Release tag names follow the pattern `v<release version>`. For example, `v1.13.1`.
Find them [here](https://github.com/microsoft/onnxruntime/tags).

## Example build commands

### Build on windows, with reduced operator support, and support for ORT format models only

```
<ONNX Runtime repository root>\build.bat ^
  --config=Release ^
  --cmake_generator="Visual Studio 16 2019" ^
  --build_shared_lib ^
  --minimal_build ^
  --disable_ml_ops --disable_exceptions --disable_rtti ^
  --include_ops_by_config <config file from model conversion> --enable_reduced_operator_type_support ^
  --skip_tests
```

### Linux

```
<ONNX Runtime repository root>/build.sh \
  --config=Release \
  --build_shared_lib \
  --minimal_build \
  --disable_ml_ops --disable_exceptions --disable_rtti \
  --include_ops_by_config <config file from model conversion> --enable_reduced_operator_type_support \
  --skip_tests
```

## Custom build packages

In this section, `ops.config` is a [configuration file](../reference/operators/reduced-operator-config-file.md) that specifies the opsets, op kernels, and types to include. See the configuration file used by the pre-built mobile packages at [tools/ci_build/github/android/mobile_package.required_operators.config](https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/github/android/mobile_package.required_operators.config).

### Web

_[This section is coming soon]_

### iOS

To produce pods for an iOS build, use the [build_and_assemble_ios_pods.py](https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/github/apple/build_and_assemble_ios_pods.py) script from the ONNX Runtime repo.

1. Check out the version of ONNX Runtime you want to use.

2. Run the build script.

    For example:

    ```bash
    python3 tools/ci_build/github/apple/build_and_assemble_ios_pods.py \
      --staging-dir /path/to/staging/dir \
      --include-ops-by-config /path/to/ops.config \
      --build-settings-file /path/to/build_settings.json
    ```

    This will do a custom build and create the pod package files for it in /path/to/staging/dir.

    The build options are specified with the file provided to the `--build-settings-file` option. See the current build options used by the pre-built mobile package at [tools/ci_build/github/apple/default_mobile_ios_framework_build_settings.json](https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/github/apple/default_mobile_ios_framework_build_settings.json). You can use this file directly.

    The reduced set of ops in the custom build is specified with the file provided to the `--include_ops_by_config` option. See the current op config used by the pre-built mobile package at [tools/ci_build/github/android/mobile_package.required_operators.config](https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/github/android/mobile_package.required_operators.config) (Android and iOS pre-built mobile packages share the same config file). You can use this file directly.

    The default package does not include the training APIs. To create a training package, add `--enable_training_apis` in the build options file provided to `--build-settings-file` and add the `--variant Training` option when calling `build_and_assemble_ios_pods.py`.
    
    For example:
    
    ```bash
    # /path/to/build_settings.json is a file that includes the `--enable_training_apis` option
    
    python3 tools/ci_build/github/apple/build_and_assemble_ios_pods.py \
      --staging-dir /path/to/staging/dir \
      --include-ops-by-config /path/to/ops.config \
      --build-settings-file /path/to/build_settings.json \
      --variant Training
    ```

3. Use the local pods.

    For example, update the Podfile to use the local onnxruntime-mobile-objc pod instead of the released one:

    ```diff
    -  pod 'onnxruntime-mobile-objc'
    +  pod 'onnxruntime-mobile-objc', :path => "/path/to/staging/dir/onnxruntime-mobile-objc"
    +  pod 'onnxruntime-mobile-c', :path => "/path/to/staging/dir/onnxruntime-mobile-c"
    ```

    Note: The onnxruntime-mobile-objc pod depends on the onnxruntime-mobile-c pod. If the released onnxruntime-mobile-objc pod is used, this dependency is automatically handled. However, if a local onnxruntime-mobile-objc pod is used, the local onnxruntime-mobile-c pod that it depends on also needs to be specified in the Podfile.

### Android

To produce an Android AAR package, use the [build_custom_android_package.py](https://github.com/microsoft/onnxruntime/blob/main/tools/android_custom_build/build_custom_android_package.py) script from the ONNX Runtime repo.

The script can be used from within the repo or outside of it. Copy its [containing directory](https://github.com/microsoft/onnxruntime/blob/main/tools/android_custom_build) for usage outside of the repo.

Note: In the steps below, replace `<ORT version>` with the ONNX Runtime version you want to use, e.g., `1.13.1`.

1. Run the build script.

    For example:

    ```bash
    python3 tools/android_custom_build/build_custom_android_package.py \
      --onnxruntime_branch_or_tag v<ORT version> \
      --include_ops_by_config /path/to/ops.config \
      --build_settings /path/to/build_settings.json \
      /path/to/working/dir
    ```

    This will do a custom build and create the Android AAR package for it in `/path/to/working/dir`.

    Specify the ONNX Runtime version you want to use with the `--onnxruntime_branch_or_tag` option. The script uses a separate copy of the ONNX Runtime repo in a Docker container so this is independent from the containing ONNX Runtime repo's version.

    The build options are specified with the file provided to the `--build_settings` option. See the current build options used by the pre-built mobile package at [tools/ci_build/github/android/default_mobile_aar_build_settings.json](https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/github/android/default_mobile_aar_build_settings.json).
    
    The reduced set of ops in the custom build is specified with the file provided to the `--include_ops_by_config` option. See the current op config used by the pre-built mobile package at [tools/ci_build/github/android/mobile_package.required_operators.config](https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/github/android/mobile_package.required_operators.config).

    The `--build_settings` and `--include_ops_by_config` options are both optional and will default to what is used to build the pre-built mobile package. Not specifying either will result in a package like the pre-built mobile package.

2. Use the local custom Android AAR package.

    For example, in an Android Studio project:

    a. Copy the AAR file from `/path/to/working/dir/output/aar_out/<build config, e.g., Release>/com/microsoft/onnxruntime/onnxruntime-mobile/<ORT version>/onnxruntime-mobile-<ORT version>.aar` to the project's `<module name, e.g., app>/libs` directory.

    b. Update the project's `<module name>/build.gradle` file dependencies section:

    ```diff
    -    implementation 'com.microsoft.onnxruntime:onnxruntime-mobile:latest.release'
    +    implementation files('libs/onnxruntime-mobile-<ORT version>.aar')
    ```

### Python

If you wish to use the ONNX Runtime python bindings with a minimal build, exceptions must be enabled due to Python requiring them.

Remove `--disable_exceptions` and add `--build_wheel` to the build command in order to build a Python Wheel with the ONNX Runtime bindings.

A .whl file will be produced in the build output directory under the `<config>/dist` folder.

* The Python Wheel for a Windows Release build using build.bat would be in `<ONNX Runtime repository root>\build\Windows\Release\Release\dist\`
* The Python Wheel for a Linux Release build using build.sh would be in `<ONNX Runtime repository root>/build/Linux/Release/dist/`

The wheel can be installed using `pip`. Adjust the following command for your platform and the whl filename.

```
pip install -U .\build\Windows\Release\Release\dist\onnxruntime-1.7.0-cp37-cp37m-win_amd64.whl
```
