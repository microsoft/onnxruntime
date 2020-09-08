# ONNX Runtime for Mobile Platforms

## Overview

ONNX Runtime now supports an internal model format to minimize the build size for usage in mobile and embedded scenarios. An ONNX model can be converted to an internal ONNX Runtime format ('ORT format model') using the below instructions.

The minimal build can be used with any ORT format model, provided that the kernels for the operators used in the model were included in the build. 
  i.e. the custom build provides a set of kernels, and if that set satisfies a given ORT format model's needs, the model can be loaded and executed. 

## Steps to create model and minimal build

You will need a script from the the ONNX Runtime repository, and to also perform a custom build, so you will need to clone the repository locally. See [here](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#prerequisites) for initial steps.

The directory the ONNX Runtime repository was cloned into is referred to as `<ONNX Runtime repository root>` in this documentation.

Once you have cloned the repository, perform the following steps to create a minimal build of ONNX Runtime that is model specific:

### 1. Create ORT format model

We will use a helper python script to convert an existing ONNX format model into an ORT format model.
This will require the standard ONNX Runtime python package to be installed.
A single model is converted at a time by this script.

  - Install the ONNX Runtime nightly python package from https://test.pypi.org/project/ort-nightly/
    - e.g. `pip install -i https://test.pypi.org/simple/ ort-nightly`
    - ensure that any existing ONNX Runtime python package was uninstalled first, or use `-U` with the above command to upgrade an existing package
    - using the nightly package is temporary until ONNX Runtime version 1.5 is released
  - Convert the ONNX model to ORT format
    - `python <ONNX Runtime repository root>/tools/python/convert_onnx_model_to_ort.py <path to .onnx model>`
    - This script will first optimize the ONNX model and save it with a '.optimized.onnx' file extension
      - *IMPORTANT* this optimized ONNX model should be used as the input to the minimal build. Do NOT use the original ONNX model for that step.
    - It will next convert the optimized ONNX model to ORT format and save the file using '.ort' as the file extension.

Example:

Running `python <ORT repository root>/tools/python/convert_onnx_model_to_ort.py /models/ssd_mobilenet.onnx`
  - Will create `/models/ssd_mobilenet.optimized.onnx`, which is an ONNX format model that ONNX Runtime has optimized 
    - e.g. constant folding will have run
  - Will use `/models/ssd_mobilenet.optimized.onnx` to create `/models/ssd_mobilenet.ort` 
    - ssd_mobilenet.ort is the ORT format version of the optimized model. 


### 2. Setup information to reduce build to minimum set of operator kernels required

In order to reduce the operator kernels included in the build, the required set must be either inferred from one or more ONNX models, or explicitly specified via configuration.

To infer, put one or more optimized ONNX models in a directory. The directory will be recursively searched for '.onnx' files. 
If taking this approach (vs. creating a configuration file), you should only include the optimized ONNX models and not both the original and optimized models, as there may be kernels that are were required in the original model that are not required in the optimized model.

Alternatively a configuration file can be created to specify the set of kernels to include. 

See the documentation on the [Reduced Operator Kernel build](Reduced_Operator_Kernel_build.md) for more information. 

This step can be run prior to building, or as part of the minimal build.

#### Example usage:

##### Pre-build

Place the optimized ONNX model/s (files with '.optimized.onnx' from the 'Create ORT format model' step above) in a directory. 

Run the script to exclude unused kernels using this directory.

`python <ONNX Runtime repository root>/tools/ci_build/exclude_unused_ops.py --model_path <directory with optimized ONNX model/s>`

##### When building

When building as per the below instructions, add `--include_ops_by_model <directory with optimized ONNX model/s>` to the build command.


### 3. Create the minimal build

You will need to build ONNX Runtime from source to reduce the included operator kernels and other aspects of the binary. 

See [here](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#start-baseline-cpu) for the general ONNX Runtime build instructions. 

#### Binary size reduction options:

The follow options can be used to reduce the build size. Enable all options that your scenario allows. 

  - Enable minimal build (`--minimal_build`)
    - A minimal build will ONLY support loading and executing ORT format models. 
      - RTTI is disabled by default in this build, so adding the `--disable_rtti` build flag is not necessary.

  - Disable exceptions (`--disable_exceptions`)
    - Disables support for exceptions in the build.
      - Any locations that would have thrown an exception will instead log the error message and call abort(). 
      - Requires `--minimal_build`.
      - NOTE: This is not a valid option if you need the Python bindings (`--build_wheel`) as the Python Wheel requires exceptions to be enabled.
    - Exceptions are only used in ORT for exceptional things. If you have validated the input to be used, and validated that the model can be loaded, it is unlikely that ORT would throw an exception unless there's a system level issue (e.g. out of memory). 

  - ML op support (`--disable_ml_ops`)
    - Whilst the operator kernel reduction script will disable all unused ML operator kernels, additional savings can be achieved by removing support for ML specific types. If you know that your model has no ML ops, or no ML ops that use the Map type, this flag can be provided. 
    - See the specs for the [ONNX ML Operators](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md) if unsure.

#### Build Configuration 

The `MinSizeRel` configuration will produce the smallest binary size.
The `Release` configuration could also be used if you wish to prioritize performance over binary size.

#### Example build commands

##### Windows

`<ONNX Runtime repository root>\build.bat --config=MinSizeRel --cmake_generator="Visual Studio 16 2019" --build_shared_lib --minimal_build --disable_ml_ops --disable_exceptions`

##### Linux

`<ONNX Runtime repository root>/build.sh --config=MinSizeRel --build_shared_lib --minimal_build --disable_ml_ops --disable_exceptions`

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

## Limitations

A minimal build has the following limitations currently:
  - No support for ONNX format models
    - Model must be converted to ORT format
  - No support for runtime optimizations
    - Optimizations should be performed prior to conversion to ORT format
  - No support for runtime partitioning (assigning nodes in a model to an execution provider)
    - Execution providers that will be used at runtime must be enabled when creating the ORT format model
  - Only supports execution providers that have statically registered kernels
    - e.g. ORT CPU and CUDA execution providers
    - Execution providers that dynamically compile nodes in the graph into custom kernels at runtime are not supported
  - No support for custom operators

