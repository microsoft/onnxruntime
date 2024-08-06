---
title: Model Usability Checker
descriptions: ORT Mobile model usability checker.
parent: ORT Mobile Model Export Helpers
grand_parent: Deploy on Mobile
nav_order: 1

---
# Model Usability Checker
{: .no_toc }

The model usability checker analyzes an ONNX model regarding its suitability for usage with ORT Mobile, NNAPI and CoreML.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Usage

```
python -m onnxruntime.tools.check_onnx_model_mobile_usability --help
usage: check_onnx_model_mobile_usability.py [-h] [--log_level {debug,info}] model_path

Analyze an ONNX model to determine how well it will work in mobile scenarios.

positional arguments:
  model_path            Path to ONNX model to check

optional arguments:
  -h, --help            show this help message and exit
  --log_level {debug,info}
                        Logging level (default: info)
```

## Use with NNAPI and CoreML

The script will check if the operators in the model are supported by ORT's NNAPI Execution Provider (EP) and CoreML EP. Depending on how many operators are supported, and where they are in the model, it will estimate if using NNAPI or CoreML is likely to be beneficial. It is always recommended to performance test to validate.

Example output from this check looks like:
```
INFO:  Checking resnet50-v1-7.onnx for usability with ORT Mobile.
INFO:  Checking NNAPI
INFO:  1 partitions with a total of 121/122 nodes can be handled by the NNAPI EP.
INFO:   Partition sizes: [121]
INFO:  Unsupported nodes due to operator=0
INFO:   Caveats that have not been checked and may result in a node not actually being supported:
     ai.onnx:Conv:Only 2D Conv is supported. Weights and bias should be constant.
     ai.onnx:Gemm:If input B is not constant, transB should be 1.
     ai.onnx:GlobalAveragePool:Only 2D Pool is supported.
     ai.onnx:MaxPool:Only 2D Pool is supported.
INFO:  Unsupported nodes due to input having a dynamic shape=1
INFO:  NNAPI should work well for this model as there is one partition covering 99.2% of the nodes in the model.
INFO:  Model should perform well with NNAPI as is: YES
```

If the model has dynamic input shapes an additional check is made to estimate whether making the shapes of fixed size would help. See [onnxruntime.tools.make_dynamic_shape_fixed](./make-dynamic-shape-fixed.md) for more information. 

Example output from this check:

```
INFO:  Checking resnet50-v1-7.onnx for usability with ORT Mobile.
...
INFO:  Checking CoreML MLProgram
INFO:  2 partitions with a total of 120/122 nodes can be handled by the CoreML MLProgram EP.
INFO:   Partition sizes: [119, 1]
INFO:  Unsupported nodes due to operator=1
INFO:   Unsupported ops: ai.onnx:Flatten
INFO:   Caveats that have not been checked and may result in a node not actually being supported:
     ai.onnx:Conv:Only 1D/2D Conv is supported. Bias if provided must be constant.
     ai.onnx:Gemm:Input B must be constant.
     ai.onnx:GlobalAveragePool:Only 2D Pool is supported currently. 3D and 5D support can be added if needed.
     ai.onnx:MaxPool:Only 2D Pool is supported currently. 3D and 5D support can be added if needed.
INFO:  Unsupported nodes due to input having a dynamic shape=1
INFO:  CoreML MLProgram can be considered for this model as there are two partitions covering 98.4% of the nodes. Performance testing is required to validate.
INFO:  Model should perform well with CoreML MLProgram as is: MAYBE
INFO:  --------
INFO:  Checking if model will perform better if the dynamic shapes are fixed...
INFO:  Partition information if the model was updated to make the shapes fixed:
INFO:  2 partitions with a total of 121/122 nodes can be handled by the CoreML MLProgram EP.
INFO:   Partition sizes: [120, 1]
INFO:  Unsupported nodes due to operator=1
INFO:   Unsupported ops: ai.onnx:Flatten
INFO:   Caveats that have not been checked and may result in a node not actually being supported:
     ai.onnx:Conv:Only 1D/2D Conv is supported. Bias if provided must be constant.
     ai.onnx:Gemm:Input B must be constant.
     ai.onnx:GlobalAveragePool:Only 2D Pool is supported currently. 3D and 5D support can be added if needed.
     ai.onnx:MaxPool:Only 2D Pool is supported currently. 3D and 5D support can be added if needed.
INFO:  CoreML MLProgram can be considered for this model as there are two partitions covering 99.2% of the nodes. Performance testing is required to validate.
INFO:  Model should perform well with CoreML MLProgram if modified to have fixed input shapes: MAYBE
INFO:  Shapes can be altered using python -m onnxruntime.tools.make_dynamic_shape_fixed
```

There is diagnostic output that provides in-depth information on why the recommendations were made.

This includes
- information on individual operators that are supported or unsupported by the NNAPI and CoreML EPs
- information on how many groups (a.k.a. partitions) the supported operators are broken into
  - the more groups the worse performance will be as we have to switch between the NPU (Neural Processing Unit) and CPU each time we switch between a supported and unsupported group of nodes

## Recommendation

Finally, the script will provide a recommendation on what EP to use.

```
INFO:  As NNAPI or CoreML may provide benefits with this model it is recommended to compare the performance of the model using the NNAPI EP on Android, and the CoreML EP on iOS, against the performance using the CPU EP.
```

