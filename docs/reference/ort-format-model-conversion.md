---
title: Convert ONNX model to ORT format
description: Convert an ONNX model to ORT format to run on mobile or web
parent: Reference
has_children: false
nav_order: 4
redirect_from: /docs/tutorials/mobile/model-conversion
---

# Convert ONNX model to ORT format

## Contents
{: .no_toc}

* TOC
{:toc}

## Overview

ONNX models are converted to ORT format using the `convert_onnx_models_to_ort` script.

The conversion script performs two functions:

  1. Loads and optimizes ONNX format models, and saves them in ORT format
  2. Determines the operators, and optionally data types, required by the optimized models, and saves these in a configuration file for use in a reduced operator build, if required

The conversion script can run on a single ONNX model, or a directory. If run against a directory, the directory will be recursively searched for '.onnx' files to convert.

Each '.onnx' file is loaded, optimized, and saved in ORT format as a file with the '.ort' extension in the same location as the original '.onnx' file.

A [build configuration file](reduced-operator-config-file.md) ('required_operators.config') is produced with the operators required by the optimized ONNX models.

If [type reduction](#enable-type-reduction) is enabled (ONNX Runtime version 1.7 or later) the configuration file will also include the required types for each operator, and be called 'required_operators_and_types.config'.

If you are using a pre-built ONNX Runtime [iOS](../install.md#ios), [Android](../install.md#android) or [web](../install.md#web) package], the build configuration file is not used and can be ignored.

Conversion of ONNX format models to ORT format utilizes the ONNX Runtime python package, as the model is loaded into ONNX Runtime and optimized as part of the conversion process.

For ONNX Runtime version 1.8 and later the conversion script is run directly from the ONNX Runtime python package.

For earlier versions, the conversion script is run from the local ONNX Runtime repository.

## Installation

Install the onnxruntime python package from [https://pypi.org/project/onnxruntime/](https://pypi.org/project/onnxruntime/) in order to convert models from ONNX format to the internal ORT format. Version 1.5.3 or higher is required.

### Install the latest release

```bash
pip install onnxruntime
```

### Install a previous release

If you are building ONNX Runtime from source (custom, reduced or minimal builds), you must match the python package version to the branch of the ONNX Runtime repository you checked out.

For example, to use the 1.7 release:

```bash
git checkout rel-1.7.2
pip install onnxruntime==1.7.2
```

If you are using the `master` branch in the git repository you should use the nightly ONNX Runtime python package

```bash
pip install -U -i https://test.pypi.org/simple/ ort-nightly
```

## Script usage

ONNX Runtime version 1.8 or later:

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort <onnx model file or dir>
```

where:

* onnx mode file or dir is a path to .onnx file or directory containing one or more .onnx models

The current optional arguments are available by running the script with the `--help` argument.
Supported arguments and defaults differ slightly across ONNX Runtime versions.

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort --help
```

```output
usage: convert_onnx_models_to_ort.py [-h] [--use_nnapi]
      [--optimization_level {disable,basic,extended,all}]
      [--enable_type_reduction]
      [--custom_op_library CUSTOM_OP_LIBRARY] [--save_optimized_onnx_model]
      model_path_or_dir

Convert the ONNX model/s in the provided directory to ORT format models. 
All files with a `.onnx` extension will be processed. For each one, an ORT format model will be created in the same directory. 
A configuration file will also be created called `required_operators.config`, and will contain the list of required operators for all converted models. 
This configuration file should be used as input to the minimal build via the `--include_ops_by_config` parameter.

positional arguments:
  model_path_or_dir     Provide path to ONNX model or directory containing ONNX model/s to convert. All files with a .onnx extension, including in subdirectories, will be processed.

optional arguments:
  -h, --help            show this help message and exit
  --optimization_level {disable,basic,extended,all}
                        Level to optimize ONNX model with, prior to converting to ORT format model. 
                        These map to the onnxruntime.GraphOptimizationLevel values. 
                        If the level is 'all' the NCHWc transformer is manually disabled as it contains device specific logic, 
                        so the ORT format model must be generated on the device it will run on. 
                        Additionally, the NCHWc optimizations are not applicable to ARM devices.
  --enable_type_reduction
                        Add operator specific type information to the configuration file to potentially 
                        reduce the types supported by individual operator implementations.
  --custom_op_library CUSTOM_OP_LIBRARY
                        Provide path to shared library containing custom operator kernels to register.
  --save_optimized_onnx_model
                        Save the optimized version of each ONNX model. This will have the same optimizations 
                        applied as the ORT format model.
```

### Optional script arguments

#### Optimization level

Set the optimization level that ONNX Runtime will use to optimize the model prior to saving in ORT format.

For ONNX Runtime version 1.8 and later, *all* is recommended if the model will be run with the CPU Execution Provider (EP).

For earlier versions, *extended* is recommended, as the *all* level previously included device specific optimizations that would limit the portability of the model.

If the model is to be run with the NNAPI EP or CoreML EP, it is recommended to create an ORT format model using the *basic* optimization level. Performance testing should be done to compare running this model with the NNAPI or CoreML EP enabled vs. running the model optimized to a higher level using the CPU EP to determine the optimal setup.

See the documentation on [performance tuning mobile scenarios](../../performance/mobile-performance-tuning.md) for more information.

#### Enable type reduction

With ONNX Runtime version 1.7 and later it is possible to limit the data types the required operators support to further reduce the build size. This pruning is referred to as "operator type reduction" in this documentation. As the ONNX model/s are converted, the input and output data types required by each operator are accumulated and included in the configuration file.

If you wish to enable operator type reduction, the [Flatbuffers](https://google.github.io/flatbuffers/) python package must be installed.

```bash
pip install flatbuffers
```

For example, the ONNX Runtime kernel for Softmax supports both float and double. If your model/s uses Softmax but only with float data, we can exclude the implementation that supports double to reduce the kernel's binary size.

#### Custom Operator support

If your ONNX model uses [custom operators](../../reference/operators/add-custom-op.md), the path to the library containing the custom operator kernels must be provided so that the ONNX model can be successfully loaded. The custom operators will be preserved in the ORT format model.

#### Save optimized ONNX model

Add this flag to save the optimized ONNX model. The optimized ONNX model contains the same nodes and initializers as the ORT format model, and can be viewed in [Netron](https://netron.app/) for debugging and performance tuning.

## Previous versions of ONNX Runtime

Prior to ONNX Runtime version 1.7, the model conversion script must be run from a cloned source repository:

```bash
python <ONNX Runtime repository root>/tools/python/convert_onnx_models_to_ort.py <onnx model file or dir>
```
