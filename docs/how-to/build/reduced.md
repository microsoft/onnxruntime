---
title: Build with reduced size
grand_parent: How to
parent: Build ORT
nav_order: 5
---

# Build ORT with reduced size
{: .no_toc }

For applications where package binary size is important, ONNX Runtime provides options to reduce the build size with some functional trade-offs.

To reduce the compiled binary size of ONNX Runtime, the operator kernels included in the build can be reduced to just the kernels required by your model/s.

For deployment on mobile devices specifically, please read more detailed guidance on [How to: Build for mobile](./mobile.md).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Feature Summary

A configuration file must be created with details of the kernels that are required.

Following that, ORT must be manually built, providing the configuration file in the `--include_ops_by_config` parameter. The build process will update the ORT kernel registration source files to exclude the unused kernels.

When building ORT with a reduced set of kernel registrations, `--skip_tests` **MUST** be specified as the kernel reduction will render many of the unit tests invalid.

NOTE: The operator exclusion logic when building with an operator reduction configuration file will only disable kernel registrations each time it runs. It will NOT re-enable previously disabled kernels. If you wish to change the list of kernels included, it is best to revert the repository to a clean state (e.g. via `git reset --hard`) before building ORT again.

## Creating a configuration file with the required kernels

The script in `<ORT Root>/tools/python/create_reduced_build_config.py` should be used to create the configuration file. This file can be manually edited as needed. The configuration can be created from either ONNX or ORT format models.

```
create_reduced_build_config.py --help
usage: Script to create a reduced build config file from ONNX or ORT format model/s. [-h] [-f {ONNX,ORT}] [-t] model_path_or_dir config_path

positional arguments:
  model_path_or_dir     Path to a single model, or a directory that will be recursively searched for models to process.
  config_path           Path to write configuration file to.

optional arguments:
  -h, --help            show this help message and exit
  -f {ONNX,ORT}, --format {ONNX,ORT}
                        Format of model/s to process. (default: ONNX)
  -t, --enable_type_reduction
                        Enable tracking of the specific types that individual operators require. Operator implementations MAY support limiting the type support included
                        in the build to these types. Only possible with ORT format models. (default: False)
```

### Type reduction
{: .no_toc }

If the configuration file is created using ORT format models, the input/output types that individual operators require can be tracked if `--enable_type_reduction` is specified. This can be used to further reduce the build size if `--enable_reduced_operator_type_support` is specified when building ORT.

ONNX format models are not guaranteed to include the required per-node type information, so cannot be used with this option.

## Configuration file format

The basic format of the operator reduction configuration file is `<operator domain>;<opset for domain>;<op1>[,op2]...`

e.g.
```
#domain;opset;op1,op2...
ai.onnx;12;Add,Cast,Concat,Squeeze
```

The opset can match either the opset import for each model, or the initial ONNX opset that the operator version was first available in. If manually editing the configuration file, using the opset import value from the model is simplest.

e.g. if a model imports opset 12 of ONNX, all ONNX operators in that model can be listed under opset 12 for the 'ai.onnx' domain.

[Netron](https://netron.app/) can be used to view an ONNX model properties to discover the opset imports.
Additionally, the ONNX operator specs for [DNN](https://github.com/onnx/onnx/blob/master/docs/Operators.md) and [traditional ML](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md) operators list the individual operator versions.

### Type reduction format
{: .no_toc }

If the types an operator implementation supports can be limited to a specific set of types, this is specified in a JSON string immediately after the operator name in the configuration file.

**It is highly recommended that you first generate the configuration file using ORT format models with type reduction enabled in order to see which operators support type reduction, and how the entry is defined for the individual operators.**

The required types are generally listed per input and/or output of the operator. The type information is in a map, with 'inputs' and 'outputs' keys. The value for 'inputs' or 'outputs' is a map between the index number of the input/output and the required list of types.

For example, both the input and output types are relevant to ai.onnx:Cast. Type information for input 0 and output 0 could look like this:
  `{"inputs": {"0": ["float", "int32_t"]}, "outputs": {"0": ["float", "int64_t"]}}`

which is added directly after the operator name in the configuration file.
e.g.
  `ai.onnx;12;Add,Cast{"inputs": {"0": ["float", "int32_t"]}, "outputs": {"0": ["float", "int64_t"]}},Concat,Squeeze`

If, for example, the types of inputs 0 and 1 were important, the entry may look like this (e.g. ai.onnx:Gather):
  `{"inputs": {"0": ["float", "int32_t"], "1": ["int32_t"]}}`

Finally some operators do non-standard things and store their type information under a 'custom' key.
ai.onnx.OneHot is an example of this, where the three input types are combined into a triple.
  `{"custom": [["float", "int64_t", "int64_t"], ["int64_t", "std::string", "int64_t"]]}`

For these reasons, it is best to generate the configuration file first, and manually edit any entries if needed.