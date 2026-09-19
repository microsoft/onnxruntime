# ONNX Runtime Reduced Operator Kernel build

In order to reduce the compiled binary size of ONNX Runtime (ORT), the operator kernels included in the build can be reduced to just the kernels required by your model/s.

A configuration file must be created with details of the kernels that are required.

Following that, ORT must be manually built, providing the configuration file in the [build.py](../tools/ci_build/build.py) `--include_ops_by_config` argument.

See the [build instructions](https://www.onnxruntime.ai/docs/how-to/build.html#build-instructions) for more details on building ORT.

When building ORT with a reduced set of kernel registrations, `--skip_tests` **MUST** be specified as the kernel reduction will render many of the unit tests invalid.

The build process will generate updated ORT kernel registration and type reduction source files to exclude unused kernel implementations.
The generated files will be under the build directory and the original source files that they are based on are not directly modified.
When building, the generated files will be used instead of the original files.

The operator exclusion logic only runs during the build file generation (or "update") phase of the build process, i.e., when invoking build.py with no build phase arguments or explicitly with `--update`.

Note: It is also possible to run the operator exclusion logic independently with [reduce_op_kernels.py](../tools/ci_build/reduce_op_kernels.py). This may be useful when building ORT without using build.py.
As the generated files will go into a build directory, the build directory must be provided with the reduce_op_kernels.py `--cmake_build_dir` argument.
Note that this argument is slightly different from the build.py `--build_dir` argument - build.py will append an additional directory for the build configuration to its `--build_dir` value to get the equivalent of `--cmake_build_dir`.

## Creating a configuration file with the required kernels

The [create_reduced_build_config.py](../tools/python/create_reduced_build_config.py) script should be used to create the configuration file. This file can be manually edited as needed. The configuration can be created from either ONNX or ORT format models.

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

If the configuration file is created using ORT format models, the input/output types that individual operators require can be tracked if the `--enable_type_reduction` argument is specified. This can be used to further reduce the build size if the build.py `--enable_reduced_operator_type_support` argument is specified when building ORT.

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
Additionally, the ONNX operator specs for [DNN](https://github.com/onnx/onnx/blob/main/docs/Operators.md) and [traditional ML](https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md) operators list the individual operator versions.

### Type reduction format

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
