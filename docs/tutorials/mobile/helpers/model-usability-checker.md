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
usage: check_onnx_model_mobile_usability.py [-h] [--config_path CONFIG_PATH] [--log_level {debug,info,warning,error}] model_path

Analyze an ONNX model to determine how well it will work in mobile scenarios, and whether it is likely to be able to use the pre-built ONNX Runtime Mobile Android or iOS package.

positional arguments:
  model_path            Path to ONNX model to check

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to required operators and types configuration used to build the pre-built ORT mobile package. (default:
                        <onnxruntime package install location>\tools\mobile_helpers\mobile_package.required_operators.config)
  --log_level {debug,info,warning,error}
                        Logging level (default: info)
```

## Use with NNAPI and CoreML

The script will check if the operators in the model are supported by ORT's NNAPI Execution Provider (EP) and CoreML EP. Depending on how many operators are supported, and where they are in the model, it will estimate if using NNAPI or CoreML is likely to be beneficial. It is always recommended to performance test to validate.

Example output from this check looks like:
```
INFO:  Checking mobilenet_v1_1.0_224_quant.onnx for usability with ORT Mobile.
INFO:  Checking NNAPI
INFO:  Model should perform well with NNAPI as is: YES
INFO:  Checking CoreML
INFO:  Model should perform well with CoreML as is: NO
INFO:  Re-run with log level of DEBUG for more details on the NNAPI/CoreML issues.
```

If the model has dynamic input shapes an additional check is made to estimate whether making the shapes of fixed size would help. See [onnxruntime.tools.make_dynamic_shape_fixed](./make-dynamic-shape-fixed.md) for more information. 

Example output from this check:

```
INFO:  Checking abs_free_dimensions.onnx for usability with ORT Mobile.
INFO:  Checking NNAPI
INFO:  Model should perform well with NNAPI as is: NO
INFO:  Checking if model will perform better if the dynamic shapes are fixed...
INFO:  Model should perform well with NNAPI if modified to have fixed input shapes: YES
INFO:  Shapes can be altered using python -m onnxruntime.tools.make_dynamic_shape_fixed
```

Setting the log level to `debug` will result in significant amounts of diagnostic output that provides in-depth information on why the recommendations were made.

This includes
- information on individual operators that are supported or unsupported by the NNAPI and CoreML EPs
- information on how many groups (a.k.a. partitions) the supported operators are broken into
  - the more groups the worse performance will be as we have to switch between the NPU (Neural Processing Unit) and CPU each time we switch between a supported and unsupported group of nodes

## Use with ORT Mobile Pre-Built package

The ONNX opset and operators used in the model are checked to determine if they are supported by the ORT Mobile pre-built package.

Example output if the model can be used as-is:
```
INFO:  Checking if pre-built ORT Mobile package can be used with mobilenet_v1_1.0_224_quant.onnx once model is
       converted from ONNX to ORT format using onnxruntime.tools.convert_onnx_models_to_ort...
INFO:  Model should work with the pre-built package.
```

If the model uses an old ONNX opset, information will be provided on how to update it. 
See [onnxruntime.tools.update_onnx_opset](./index.md#onnx-model-opset-updater) for more information.

Example output:
```
INFO:  Checking if pre-built ORT Mobile package can be used with abs_free_dimensions.onnx once model is converted 
       from ONNX to ORT format using onnxruntime.tools.convert_onnx_models_to_ort...
INFO:  Model uses ONNX opset 9.
INFO:  The pre-built package only supports ONNX opsets [12, 13, 14, 15].
INFO:  Please try updating the ONNX model opset to a supported version using
       python -m onnxruntime.tools.onnx_model_utils.update_onnx_opset ...
```

## Recommendation

Finally the script will provide information on how to [convert the model to the ORT format](../../../../docs/performance/model-optimizations/ort-format-models.md) required by ORT Mobile, and recommend which of the two ORT format models to use.

```
INFO:  Run `python -m onnxruntime.tools.convert_onnx_models_to_ort ...` to convert the ONNX model to ORT format. 
       By default, the conversion tool will create an ORT format model with saved optimizations which can potentially be 
       applied at runtime (with a .with_runtime_opt.ort file extension) for use with NNAPI or CoreML, and a fully
       optimized ORT format model (with a .ort file extension) for use with the CPU EP.
INFO:  As NNAPI or CoreML may provide benefits with this model it is recommended to compare the performance of 
       the <model>.with_runtime_opt.ort model using the NNAPI EP on Android, and the CoreML EP on iOS, against the 
       performance of the <model>.ort model using the CPU EP.
```

