# ONNX Runtime Reduced Operator Kernel build

In order to reduce the compiled binary size of ONNX Runtime (ORT), the operator kernels included in the build can be reduced to just the kernels required for your scenario. 

The kernels to include must first be identified, and secondly the ORT kernel registration source files must be updated to exclude the unused kernels. 

Finally ORT must be manually built.

When building ORT with a reduced set of kernel registrations, `--skip_tests` *MUST* be specified as the kernel reduction will render many of the unit tests invalid. 

## Selecting Required Kernels

Two options are available for selecting the required operator kernels. These options can be combined.

### Selection via ONNX models

Put the ONNX model/s you wish to be able to execute with a reduced version of ORT in a directory. The selection script will recursively look for all '.onnx' models in this directory, and aggregate information on the kernels required. 

### Selection via configuration file

A configuration file can also be used to specify the required kernels. 
The format is `<operator domain>;<opset for domain>;<op>[,op]...`

The opset should match the opset import for each model. It does not need to match the initial ONNX opset that the operator was available in. 
e.g. if a  model imports opset 12 of ONNX, all ONNX operators in that model should be listed under opset 12 for the 'ai.onnx' domain.

Example config that could be used for a scenario with 2 simplistic models. One targeting ONNX opset 10 with an Add and Concat node, the other targeting ONNX opset 12 with an Add and Split node.

```
#domain;opset;op1,op2...
ai.onnx;10;Add,Concat
ai.onnx;12;Add,Split
```

## Reducing Build to Required Kernels

There are two ways to reduce the kernels included in the build to the required ones.
  - via build script arguments when building ORT
    - the exclusion script will be run as part of the build process
  - via directly running the exclusion script prior to building ORT

NOTE: The exclusion script will only disable kernel registrations each time it runs. It will NOT re-enable previously disabled kernels. If you wish to change the list of kernels to include it is best to revert the repository to a clean state (`git reset --hard`) before running either the ORT build script or the exclusion script each time.

### Build time reduction

When running the ORT build script there are two arguments that can be used. These may be combined. 

  - `--include_ops_by_model=<path to directory containing ONNX model/s\>`
  - `--include_ops_by_config=<path to configuration file\>`

`--skip_tests` MUST also be specified.

See the ORT [build instructions](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#build-instructions) for more details.

Most likely the `--config` value should be Release or MinSizeRel.

### Pre-build reduction

The script to reduce the kernel registrations can be found in `<ORT repository root>/tools/ci_build/exclude_unused_ops.py`.

It can be run in a similar fashion. 
`--model_path` is a path to a directory containing one or more ONNX models. Directory is recursively searched.
`--file_path` is a path to a configuration file for the required operators
`--ort_root` is the path to the ORT repository root that the kernel registration exclusions should be done in. If not provided it will default to be the repository containing the exclude_unused_ops.py script.

```
python exclude_unused_ops.py --model_path d:\ReduceSize\models --config_path d:\ReduceSize\ops_config.txt --ort_root d:\onnxruntime
```

After running the script build ORT as per the build instructions. Remember to specify `--skip-tests`.

#### Generating configuration file

Note: It is also possible to generate a configuration file for future usage by providing the `--write_combined_config_to` argument to `exclude_unused_ops.py`.
If run this way it will process the information provided by `--model_path` and/or `--config_path`, and output a configuration file with the combined list of required operators to the provided path. 
No kernel registration changes will be made when run this way. 

