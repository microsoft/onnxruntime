# ONNXRuntime Format Model Usage

## Overview

ONNXRuntime (ORT) now supports an internal ORT model format to minimize the build size for usage in mobile and embedded scenarios.

The minimal build can be used with any ORT format model, provided that the kernels for the operators used in the model were included. i.e. the custom build provides a set of kernels, and if that set satisfies a given ORT format model's needs the model can be loaded and executed. 

## Steps to create model and minimal build

You will need a script from the the ORT repository, and to perform a custom build, so you will need to clone the repository locally. See [here](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#prerequisites) for initial steps.

Perform the following steps to create a minimal build from an ONNX format model.

### Create ORT format model

We will use a helper python script to convert an existing ONNX format model into an optimized ORT format model.
This will require the ORT python package to be installed, and the ORT repository to have been cloned.

  - Install the ORT nighly python package from https://test.pypi.org/project/ort-nightly/
  - Convert the ONNX model to ORT format
    - `python <path to ORT repository>/tools/python/convert_onnx_model_to_ort.py <path to .onnx model>`
    - This script will first optimize the ONNX model and save it with a '.optimized.onnx' file extension
      - this optimized model should be used as input to the reduced build
    - It will next convert the optimized ONNX model to ORT format.

### Setup information to reduce build to minimum set of operator kernels required

In order to reduce the operator kernels included in the build the required set must be either inferred from one or more ONNX models, or explicitly specified.d

To infer, put one or more ONNX models in a directory. The directory will be recursively searched for '.onnx' files. If taking this approach, you should only include the optimized ONNX models and not both the original and optimized models.

Alternatively a configuration file can be created for the set of kernels to include. 

See the documentation on the [Reduced Operator Kernel build](Reduced_Operator_Kernel_build.md) for more information. This step can be run separately or as part of the minimal build.

### Create the minimal build

You will need to build ORT from source to reduce the operator kernels and other aspects of the binary. 

See [here](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#start-baseline-cpu) for build instructions. 

Binary size reduction options.
  - Enable minimal build (`--minimal_build`)
    - A minimal build will ONLY support loading and executing ORT format models. RTTI is disabled by default in this build.

  - Disable exceptions (`--disable_exceptions`)
    - Disables exceptions in the build. Any locations that would have thrown an exception will instead log the error message and call abort(). 
        - Requires `--minimal_build`
        - Is not a valid option if you need the python bindings (`--build_wheel`) as python/pybind cannot be built with exceptions disabled.
    - Exceptions are only used in ORT for exceptional things. If you have validated the input to be used, and validated that the model can be loaded, it is unlikely that ORT would throw an exception unless there's a system level issue (e.g. out of memory). 

  - ML op support (`--disable_ml_ops`)
    - Whilst the operator kernel reduction script will disable all unused ML operator kernels, additional savings can be achieved by removing support for ML specific types. If you know your model has no ML ops, or no ML ops that use the Map type, this flag can be provided. 
    - See the specs for the [ONNX ML Operators](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md) if unsure.


#### Example build commands

##### Windows

`<ORT repository root>\build.bat --config=MinSizeRel --cmake_generator="Visual Studio 16 2019" --build_shared_lib --minimal_build --disable_ml_ops --disable_exceptions`

##### Linux

`<ORT repository root>/build.sh --config=MinSizeRel --build_shared_lib --minimal_build --disable_ml_ops --disable_exceptions`

## Executing ORT format models

The API for executing ORT format models is the same as for ONNX models. See the [ORT API documentation](https://github.com/Microsoft/onnxruntime/#api-documentation).

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

A minimal build has the following limitations
  - No support for ONNX format models
    - Model must be converted to ORT format
  - No support for runtime optimizations
    - Optimizations should be performed prior to conversion to ORT format
  - No support for runtime partioning
    - Execution providers that will be used at runtime must be enabled when creating the ORT format model
  - Only supports execution providers that have statically registered kernels
    - e.g. ORT CPU and CUDA execution providers
    - Execution providers that dynamically compile nodes in the graph into custom kernels at runtime are not supported
  - No support for custom operators

