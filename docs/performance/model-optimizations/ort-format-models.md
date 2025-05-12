---
title: ORT model format
description: Define the ORT format and show how to convert an ONNX model to ORT format to run on mobile or web
grand_parent: Performance
parent: Model optimizations
nav_order: 4
redirect_from: 
  - /docs/tutorials/mobile/model-conversion
  - /docs/tutorials/mobile/model-execution
  - /docs/reference/ort-format-models
---

# ORT model format
{: .no_toc}

## Contents
{: .no_toc}

* TOC
{:toc}

## What is the ORT model format?

The ORT format is the format supported by reduced size ONNX Runtime builds. Reduced size builds may be more appropriate for use in size-constrained environments such as mobile and web applications.

Both ORT format models and ONNX models are supported by a full ONNX Runtime build.

## Backwards Compatibility

Generally, the goal is that a particular version of ONNX Runtime can run models at the current (at time of the ONNX Runtime release) or older versions of the ORT format.

Though we try to maintain backwards compatibility, there have been some breaking changes.

ONNX Runtime version|ORT format version support|Notes
-|-|-
1.14+|v5, v4 (limited support)| See [here](https://github.com/microsoft/onnxruntime/blob/rel-1.14.0/docs/ORT_Format_Update_in_1.13.md#onnx-runtime-114) for details about limited v4 support.
1.13|v5|v5 breaking change: removed kernel def hashes.
1.12-1.8|v4|v4 breaking change: updated kernel def hash computation.
1.7|v3, v2, v1|
1.6|v2, v1|
1.5|v1|ORT format introduced

## Convert ONNX models to ORT format

ONNX models are converted to ORT format using the `convert_onnx_models_to_ort` script.

The conversion script performs two functions:

  1. Loads and optimizes ONNX format models, and saves them in ORT format
  2. Determines the operators, and optionally data types, required by the optimized models, and saves these in a configuration file for use in a reduced operator build, if required

The conversion script can run on a single ONNX model, or a directory. If run against a directory, the directory will be recursively searched for '.onnx' files to convert.

Each '.onnx' file is loaded, optimized, and saved in ORT format as a file with the '.ort' extension in the same location as the original '.onnx' file.

### Outputs of the script

1. One ORT format model for each ONNX model
2. A [build configuration file](../../reference/operators/reduced-operator-config-file.md) ('required_operators.config') with the operators required by the optimized ONNX models.

   If [type reduction](#enable-type-reduction) is enabled (ONNX Runtime version 1.7 or later) the configuration file will also include the required types for each operator, and is called 'required_operators_and_types.config'.

   If you are using a pre-built ONNX Runtime [iOS](../../install/index.md#install-on-ios), [Android](../../install/index.md#install-on-android) or [web](../../install/index.md#javascript-installs) package, the build configuration file is not used and can be ignored.

### Script location

The ORT model format is supported by version 1.5.2 of ONNX Runtime or later.

Conversion of ONNX format models to ORT format utilizes the ONNX Runtime python package, as the model is loaded into ONNX Runtime and optimized as part of the conversion process.

For ONNX Runtime version 1.8 and later the conversion script is run directly from the ONNX Runtime python package.

For earlier versions, the conversion script is run from the local ONNX Runtime repository.

### Install ONNX Runtime

Install the onnxruntime python package from [https://pypi.org/project/onnxruntime/](https://pypi.org/project/onnxruntime/) in order to convert models from ONNX format to the internal ORT format. Version 1.5.3 or higher is required.

#### Install the latest release

```bash
pip install onnxruntime
```

#### Install a previous release

If you are building ONNX Runtime from source (custom, reduced or minimal builds), you must match the python package version to the branch of the ONNX Runtime repository you checked out.

For example, to use the 1.7 release:

```bash
git checkout rel-1.7.2
pip install onnxruntime==1.7.2
```

If you are using the `main` branch in the git repository you should use the nightly ONNX Runtime python package:

```bash
pip install -U -i https://test.pypi.org/simple/ ort-nightly
```

### Convert ONNX models to ORT format script usage

ONNX Runtime version 1.8 or later:

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort <onnx model file or dir>
```

where:

* onnx model file or dir is a path to .onnx file or directory containing one or more .onnx models

The current optional arguments are available by running the script with the `--help` argument.
Supported arguments and defaults differ slightly across ONNX Runtime versions.

Help text from ONNX Runtime 1.11:
```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort --help
```

```output
  usage: convert_onnx_models_to_ort.py [-h] [--optimization_style {Fixed,Runtime} [{Fixed,Runtime} ...]] [--enable_type_reduction] [--custom_op_library CUSTOM_OP_LIBRARY] [--save_optimized_onnx_model] [--allow_conversion_failures] [--nnapi_partitioning_stop_ops NNAPI_PARTITIONING_STOP_OPS]
                                      [--target_platform {arm,amd64}]
                                      model_path_or_dir

  Convert the ONNX format model/s in the provided directory to ORT format models. All files with a `.onnx` extension will be processed. For each one, an ORT format model will be created in the same directory. A configuration file will also be created containing the list of required
  operators for all converted models. This configuration file should be used as input to the minimal build via the `--include_ops_by_config` parameter.

  positional arguments:
    model_path_or_dir     Provide path to ONNX model or directory containing ONNX model/s to convert. All files with a .onnx extension, including those in subdirectories, will be processed.

  optional arguments:
    -h, --help            show this help message and exit
    --optimization_style {Fixed,Runtime} [{Fixed,Runtime} ...]
                          Style of optimization to perform on the ORT format model. Multiple values may be provided. The conversion will run once for each value. The general guidance is to use models optimized with 'Runtime' style when using NNAPI or CoreML and 'Fixed' style otherwise.
                          'Fixed': Run optimizations directly before saving the ORT format model. This bakes in any platform-specific optimizations. 'Runtime': Run basic optimizations directly and save certain other optimizations to be applied at runtime if possible. This is useful when
                          using a compiling EP like NNAPI or CoreML that may run an unknown (at model conversion time) number of nodes. The saved optimizations can further optimize nodes not assigned to the compiling EP at runtime.
    --enable_type_reduction
                          Add operator specific type information to the configuration file to potentially reduce the types supported by individual operator implementations.
    --custom_op_library CUSTOM_OP_LIBRARY
                          Provide path to shared library containing custom operator kernels to register.
    --save_optimized_onnx_model
                          Save the optimized version of each ONNX model. This will have the same level of optimizations applied as the ORT format model.
    --allow_conversion_failures
                          Whether to proceed after encountering model conversion failures.
    --nnapi_partitioning_stop_ops NNAPI_PARTITIONING_STOP_OPS
                          Specify the list of NNAPI EP partitioning stop ops. In particular, specify the value of the "ep.nnapi.partitioning_stop_ops" session options config entry.
    --target_platform {arm,amd64}
                          Specify the target platform where the exported model will be used. This parameter can be used to choose between platform-specific options, such as QDQIsInt8Allowed(arm), NCHWc (amd64) and NHWC (arm/amd64) format, different optimizer level options, etc.
```

#### Optional script arguments

##### Optimization style

**Since ONNX Runtime 1.11**

Specify whether the converted model will be fully optimized ("Fixed") or have saved runtime optimizations ("Runtime"). Both types of models are produced by default. See [here](./ort-format-model-runtime-optimization.md) for more information.

This replaces the [optimization level](#optimization-level) option from earlier ONNX Runtime versions.

##### Optimization level

**ONNX Runtime version 1.10 and earlier**

Set the optimization level that ONNX Runtime will use to optimize the model prior to saving in ORT format.

For ONNX Runtime version 1.8 and later, *all* is recommended if the model will be run with the CPU EP.

For earlier versions, *extended* is recommended, as the *all* level previously included device specific optimizations that would limit the portability of the model.

If the model is to be run with the NNAPI EP or CoreML EP, it is recommended to create an ORT format model using the *basic* optimization level. Performance testing should be done to compare running this model with the NNAPI or CoreML EP enabled vs. running the model optimized to a higher level using the CPU EP to determine the optimal setup.

See the documentation on [performance tuning mobile scenarios](../../performance/mobile-performance-tuning.md) for more information.

##### Enable type reduction

With ONNX Runtime version 1.7 and later it is possible to limit the data types the required operators support to further reduce the build size. This pruning is referred to as "operator type reduction" in this documentation. As the ONNX model/s are converted, the input and output data types required by each operator are accumulated and included in the configuration file.

If you wish to enable operator type reduction, the [Flatbuffers](https://google.github.io/flatbuffers/) python package must be installed.

```bash
pip install flatbuffers
```

For example, the ONNX Runtime kernel for Softmax supports both float and double. If your model/s uses Softmax but only with float data, we can exclude the implementation that supports double to reduce the kernel's binary size.

##### Custom Operator support

If your ONNX model uses [custom operators](../../reference/operators/add-custom-op.md), the path to the library containing the custom operator kernels must be provided so that the ONNX model can be successfully loaded. The custom operators will be preserved in the ORT format model.

##### Save optimized ONNX model

Add this flag to save the optimized ONNX model. The optimized ONNX model contains the same nodes and initializers as the ORT format model, and can be viewed in [Netron](https://netron.app/) for debugging and performance tuning.

### Previous versions of ONNX Runtime

Prior to ONNX Runtime version 1.7, the model conversion script must be run from a cloned source repository:

```bash
python <ONNX Runtime repository root>/tools/python/convert_onnx_models_to_ort.py <onnx model file or dir>
```

## Load and execute a model in ORT format

The API for executing ORT format models is the same as for ONNX models.

See the [ONNX Runtime API documentation](../../api) for details on individual API usage.

### APIs by platform

| Platform | Available APIs |
|----------|----------------|
| Android | C, C++, Java, Kotlin |
| iOS | C, C++, Objective-C (Swift via bridge) |
| Web | JavaScript |

### ORT format model loading

If you provide a filename for the ORT format model, a file extension of '.ort' will be inferred to be an ORT format model.

If you provide in-memory bytes for the ORT format model, a marker in those bytes will be checked to determine if it's an ORT format model.

If you wish to explicitly say that the InferenceSession input is an ORT format model you can do so via SessionOptions, although this generally should not be necessary.

#### Load ORT format model from a file path

C++ API

```c++
Ort::SessionOptions session_options;
session_options.AddConfigEntry("session.load_model_format", "ORT");

Ort::Env env;
Ort::Session session(env, <path to model>, session_options);
```

Java API

```java
SessionOptions session_options = new SessionOptions();
session_options.addConfigEntry("session.load_model_format", "ORT");

OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession(<path to model>, session_options);
```

JavaScript API

```js
import * as ort from "onnxruntime-web";

const session = await ort.InferenceSession.create("<path to model>");
```

#### Load ORT format model from an in-memory byte array

If a session is created using an input byte array containing the ORT format model data, by default we will copy the model bytes at the time of session creation to ensure the model bytes buffer is valid.

You may also enable the option to use the model bytes directly by setting the SessionOptions config entry `session.use_ort_model_bytes_directly` to `1`.  This may reduce the peak memory usage of ONNX Runtime Mobile, but you will need to guarantee that the model bytes are valid throughout the lifespan of the ORT session. For ONNX Runtime Web, this option is set by default.

If `session.use_ort_model_bytes_directly` is enabled there is also an option to directly use the model bytes for initializers to further reduce peak memory usage. Set the Session Options config entry `session.use_ort_model_bytes_for_initializers` to `1` to enable this.
Note that if an initializer gets pre-packed it will undo the peak memory usage saving from using the model bytes directly for that initializer, as a new buffer for the pre-packed data needs to be allocated. Pre-packing is an optional performance optimization that involves changing the initializer layout to the optimal ordering for the current platform if it differs. If reducing peak memory usage is more important than potential performance optimizations pre-packing can be disabled by setting `session.disable_prepacking` to `1`.


C++ API
```c++
Ort::SessionOptions session_options;
session_options.AddConfigEntry("session.load_model_format", "ORT");
session_options.AddConfigEntry("session.use_ort_model_bytes_directly", "1");

std::ifstream stream(<path to model>, std::ios::in | std::ios::binary);
std::vector<uint8_t> model_bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

Ort::Env env;
Ort::Session session(env, model_bytes.data(), model_bytes.size(), session_options);
```

Java API
```java
SessionOptions session_options = new SessionOptions();
session_options.addConfigEntry("session.load_model_format", "ORT");
session_options.addConfigEntry("session.use_ort_model_bytes_directly", "1");

byte[] model_bytes = Files.readAllBytes(Paths.get(<path to model>));

OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession(model_bytes, session_options);
```

JavaScript API
```js
import * as ort from "onnxruntime-web";

const response = await fetch(modelUrl);
const arrayBuffer = await response.arrayBuffer();
model_bytes = new Uint8Array(arrayBuffer);

const session = await ort.InferenceSession.create(model_bytes);
```
