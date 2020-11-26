# ONNX Runtime for Mobile Platforms

## Overview

<img align="left" width=40% src="images/Mobile.png" alt="Steps to build the reduced binary size."/>

ONNX Runtime now supports an internal model format to minimize the build size for usage in mobile and embedded scenarios. An ONNX model can be converted to an internal ONNX Runtime format ('ORT format model') using the below instructions.

The minimal build can be used with any ORT format model, provided that the kernels for the operators used in the model were included in the build.
    i.e. the custom build provides a set of kernels, and if that set satisfies a given ORT format model's needs, the model can be loaded and executed.

## Steps to create model and minimal build

You will need a script from the the ONNX Runtime repository, and to also perform a custom build, so you will need to clone the repository locally. See [here](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#prerequisites) for initial steps.

The directory the ONNX Runtime repository was cloned into is referred to as `<ONNX Runtime repository root>` in this documentation.

Once you have cloned the repository, perform the following steps to create a minimal build of ONNX Runtime that is model specific:

### 1. Create ORT format model and configuration file with required operators

We will use a helper python script to convert ONNX format models into ORT format models, and to create the configuration file for use with the minimal build.
This will require the standard ONNX Runtime python package to be installed.

  - Install the ONNX Runtime nightly python package from https://test.pypi.org/project/ort-nightly/
    - e.g. `pip install -i https://test.pypi.org/simple/ ort-nightly`
    - ensure that any existing ONNX Runtime python package was uninstalled first, or use `-U` with the above command to upgrade an existing package
    - using the nightly package is temporary until ONNX Runtime version 1.5 is released
  - Copy all the ONNX models you wish to convert and use with the minimal build into a directory
  - Convert the ONNX models to ORT format 
    - `python <ONNX Runtime repository root>/tools/python/convert_onnx_models_to_ort.py <path to directory containing one or more .onnx models>`
      - For each ONNX model an ORT format model will be created with '.ort' as the file extension.
      - A `required_operators.config` configuration file will also be created.

Example:

Running `'python <ORT repository root>/tools/python/convert_onnx_models_to_ort.py /models'` where the '/models' directory contains ModelA.onnx and ModelB.onnx
  - Will create /models/ModelA.ort and /models/ModelB.ort
  - Will create /models/required_operators.config/

### 2. Create the minimal build

You will need to build ONNX Runtime from source to reduce the included operator kernels and other aspects of the binary. 

See [here](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#start-baseline-cpu) for the general ONNX Runtime build instructions. 

#### Binary size reduction options:

The follow options can be used to reduce the build size. Enable all options that your scenario allows.
  - Reduce build to required operator kernels
    - Add `--include_ops_by_config <config file produced by step 1> --skip_tests` to the build parameters.
    - See the documentation on the [Reduced Operator Kernel build](Reduced_Operator_Kernel_build.md) for more information. This step can also be done pre-build if needed.
      - NOTE: This step will edit some of the ONNX Runtime source files to exclude unused kernels. If you wish to go back to creating a full build, or wish to change the operator kernels included, you should run `git reset --hard` or `git checkout HEAD -- ./onnxruntime/core/providers` to undo these changes.

  - Enable minimal build (`--minimal_build`)
    - A minimal build will ONLY support loading and executing ORT format models. 
    - RTTI is disabled by default in this build, unless the Python bindings (`--build_wheel`) are enabled. 
    - If you wish to enable a compiling execution provider such as NNAPI specify `--minimal_build extended`. 
      - See [here](#Enabling-Execution-Providers-that-compile-kernels-in-a-minimal-build) for more information

  - Disable exceptions (`--disable_exceptions`)
    - Disables support for exceptions in the build.
      - Any locations that would have thrown an exception will instead log the error message and call abort(). 
      - Requires `--minimal_build`.
      - NOTE: This is not a valid option if you need the Python bindings (`--build_wheel`) as the Python Wheel requires exceptions to be enabled.
    - Exceptions are only used in ONNX Runtime for exceptional things. If you have validated the input to be used, and validated that the model can be loaded, it is unlikely that ORT would throw an exception unless there's a system level issue (e.g. out of memory). 

  - ML op support (`--disable_ml_ops`)
    - Whilst the operator kernel reduction script will disable all unused ML operator kernels, additional savings can be achieved by removing support for ML specific types. If you know that your model has no ML ops, or no ML ops that use the Map type, this flag can be provided. 
    - See the specs for the [ONNX ML Operators](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md) if unsure.

  - Use shared libc++ on Android (`--android_cpp_shared`)
    - Building using the shared libc++ library instead of the default static libc++ library will result in a smaller libonnxruntime.so library.
    - See [Android NDK documentation](https://developer.android.com/ndk/guides/cpp-support) for more information.

#### Build Configuration 

The `MinSizeRel` configuration will produce the smallest binary size.
The `Release` configuration could also be used if you wish to prioritize performance over binary size.

#### Example build commands

##### Windows

`<ONNX Runtime repository root>\build.bat --config=MinSizeRel --cmake_generator="Visual Studio 16 2019" --build_shared_lib --minimal_build --disable_ml_ops --disable_exceptions --include_ops_by_config <config file produced by step 1> --skip_tests`

##### Linux

`<ONNX Runtime repository root>/build.sh --config=MinSizeRel --build_shared_lib --minimal_build --disable_ml_ops --disable_exceptions --include_ops_by_config <config file produced by step 1> --skip_tests`

##### Building ONNX Runtime Python Wheel as part of Minimal build

Remove `--disable_exceptions` (Python requires exceptions to be enabled) and add `--build_wheel` to build a Python Wheel with the ONNX Runtime bindings. 
A .whl file will be produced in the build output directory under the `<config>/dist` folder.

  - The Python Wheel for a Windows MinSizeRel build using build.bat would be in `<ONNX Runtime repository root>\build\Windows\MinSizeRel\MinSizeRel\dist\`
  - The Python Wheel for a Linux MinSizeRel build using build.sh would be in `<ONNX Runtime repository root>/build/Linux/MinSizeRel/dist/`

The wheel can be installed using `pip`. Adjust the following command for your platform and the whl filename.
  -  `pip install -U .\build\Windows\MinSizeRel\MinSizeRel\dist\onnxruntime-1.4.0-cp37-cp37m-win_amd64.whl`

## Executing ORT format models

The API for executing ORT format models is the same as for ONNX models. See the [ONNX Runtime API documentation](https://github.com/Microsoft/onnxruntime/#api-documentation).

If you provide a filename for the ORT format model, a file extension of '.ort' will be inferred to be an ORT format model.
If you provide in-memory bytes for the ORT format model, a marker in those bytes will be checked to infer if it's an ORT format model.

If you wish to explicitly say that the InferenceSession input is an ORT format model you can do so via SessionOptions.

C++ API
```C++
Ort::SessionOptions session_options;
session_options.AddConfigEntry('session.load_model_format', 'ORT');
```

Python
```python
so = onnxruntime.SessionOptions()
so.add_session_config_entry('session.load_model_format', 'ORT')
session = onnxruntime.InferenceSession(<path to model>, so)
```

## Advanced Usage
### Enabling Execution Providers that compile kernels in a minimal build

It is possible to enable execution providers that compile kernels in a minimal build. 
Currently the NNAPI execution provider is the only execution provider that has support for running in a minimal build.

#### Create NNAPI aware ORT format model 
  - Create a 'full' (i.e. no usage of the `--minimal_build` flag) build of ONNX Runtime with NNAPI enabled
    - **NOTE** do this prior to creating the minimal build
      - the process for creating a minimal build will exclude operators that may be needed to load the ONNX model and create the ORT format model
      - if you have previously done a minimal build, run `git reset --hard` to make sure any operator kernel exclusions are reversed
    - we can not use the ONNX Runtime prebuilt package as NNAPI is not enabled in it
    - the 'full' build can be done on any platform
      - you do NOT need to create an Android build of ONNX Runtime in order to create an ORT format model that is optimized for usage with NNAPI.
      - when the NNAPI execution provider is enabled on non-Android platforms it can only specify which nodes can be assigned to NNAPI. it can NOT be used to execute the model.
    - perform a standard build as per the [common build instructions](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#common-build-instructions), and add `--use_nnapi --build_shared_lib --build_wheel` to the build flags if any of those are missing
  - Install the python wheel from the build output directory
    - this is located in `build/Windows/<config>/<config>/dist/<package name>.whl` on Windows, or  `build/Linux/<config>/dist/<package name>.whl` on Linux. 
      - `<config>` is the value from the `--config` parameter from the build command (e.g. Release)
      - the package name will differ based on your platform, python version, and build parameters
      - e.g. `pip install -U build\Windows\Release\Release\dist\onnxruntime-1.5.2-cp37-cp37m-win_amd64.whl`
  - Create an ORT format model by running `tools\python\convert_onnx_models_to_ort.py` as per the above instructions, with the addition of the `--use_nnapi` parameter
    - the python package from your 'full' build with NNAPI enabled must be installed for `--use_nnapi` to be a valid option
    - this will preserve all the nodes that can be assigned to NNAPI, as well as setup the ability to fallback to CPU execution if NNAPI is not available at runtime, or if NNAPI can not run all the nodes due to device limitations.

The generated ORT format model can be used on all platforms, however there is an important caveat:
  - Basic optimization such as constant folding run prior to the NNAPI execution provider being asked to nominate the nodes it can handle. These optimizations will be included in the ORT format model
  - Any potential extended optimizations on nodes that the NNAPI execution provider claims will not occur
    - these are optimizations that involve custom non-ONNX operators 
      - e.g. custom ONNX Runtime FusedConv operator that combines a Conv node and activation node (e.g. Relu)
  - Depending on the model, and how many of these potential extended optimizations are prevented, there may be some performance loss if the NNAPI execution provider is not available (e.g. running on a non-Android platform), or does not claim the same set of nodes at runtime
    - whether there is any performance loss, and/or whether there is significant performance loss, is model dependent
      - please test to ascertain what works best for your scenarios
        - you may want to generate one NNAPI aware ORT format model, and one generic ORT format model

*Side note:* If losing the extended optimizations is not a concern, you can simply generate an ORT format model that can be used with NNAPI using the default ONNX Runtime package. Specify `--optimization_level basic` instead of `--use_nnapi` when running `tools\python\convert_onnx_models_to_ort.py`. This will mean all nodes that NNAPI could potentially will handle remain available, and at runtime the NNAPI execution provider can take them.

#### Create the minimal build with NNAPI support
NOTE: A minimal build with full NNAPI support can only be for the Android platform. 
See [these](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#Android-NNAPI-Execution-Provider) instructions for details on creating an Android build with NNAPI included. 

  - Follow the [above](#2-Create-the-minimal-build) instructions to create the minimal build, with the following changes:
    - Add `--minimal_build extended` to enable the support for execution providers that compile kernels in the minimal build.
    - Add `--use_nnapi` to include NNAPI in the build

## Limitations

A minimal build has the following limitations currently:
  - No support for ONNX format models
    - Model must be converted to ORT format
  - No support for runtime optimizations
    - Optimizations should be performed prior to conversion to ORT format
  - Limited support for runtime partitioning (assigning nodes in a model to specific execution providers)
    - Execution providers that will be used at runtime MUST be enabled when creating the ORT format model
    - Execution providers that statically register kernels are supported by default (e.g. ORT CPU Execution Provider)
    - Execution providers that compile nodes are optionally supported, and nodes they create will be correctly partitioned
      - currently this is limited to the NNAPI execution provider
  - No support for custom operators

We do not currently offer backwards compatibility guarantees for ORT format models, as we will be expanding the capabilities in the short term and may need to update the internal format in an incompatible manner to accommodate these changes. You may need to regenerate the ORT format models to use with a future version of ONNX Runtime. Once the feature set stabilizes we will provide backwards compatibility guarantees.

