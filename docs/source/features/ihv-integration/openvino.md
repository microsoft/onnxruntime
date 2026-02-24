# OpenVINO

OpenVINO is a cross-platform deep learning toolkit developed by Intel. The name stands for "Open Visual Inference and Neural Network
Optimization." OpenVINO focuses on optimizing neural network inference with a write-once, deploy-anywhere approach for Intel hardware
platforms.

Read more at: [Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)

The `nncf` package (Neural Network Compression Framework) is used for model compression and quantization. It is required for workflows involving post-training quantization or other advanced optimization techniques in OpenVINO.

For Generative AI models, install Optimum Intel® from [Optimum Intel® Installation Instructions](https://huggingface.co/docs/optimum/main/en/intel/installation)

## Prerequisites

Note: OpenVINO version in Olive >= 2025.3.0

### Option 1: install Olive with OpenVINO extras

```bash
pip install olive-ai[openvino]
```

### Option 2: Install OpenVINO Runtime and OpenVINO Development Tools from Pypi

```bash
pip install openvino>=2025.4.1
pip install nncf>=2.19.0
pip install onnxruntime-openvino
```

### Install Optimum Intel® for Generative AI Workloads

```bash
pip install optimum[openvino]>=2.1.0
```

More detailed instructions are available at [Optimum Installation Instructions](https://huggingface.co/docs/optimum/installation) and at [Optimum Intel® Installation Instructions](https://huggingface.co/docs/optimum/main/en/intel/installation)

## Model Conversion

`OpenVINOConversion` pass will convert the model from original framework to OpenVINO IR Model. `PyTorchModelHandler`, `ONNXModelHandler` and
`TensorFlowModelHandler` are supported for now.

Please refer to [OpenVINOConversion](https://microsoft.github.io/Olive/reference/pass.html#openvinoconversion) for more details about the pass and its config parameters.

### Example Conversion Configuration


```json
{
    "type": "OpenVINOConversion",
    "input_shapes": [[1, 3, 32, 32]]
}
```

## Model IoUpdate

`OpenVINOIoUpdate` pass is a required pass used only for OpenVino IR Model. It converts `OpenVINOModelHandler` into a static shaped model and
to update input and output tensors.


Please refer to [OpenVINOIoUpdate](https://microsoft.github.io/Olive/reference/pass.html#openvinoioupdate) for more details about the pass and its config parameters.
The `"static"` parameter defaults to `true` and does not need to be explicitly overridden.

### Example IO Update Configuration

```json
{
    "type": "OpenVINOIoUpdate",
    "input_shapes": [[1, 3, 32, 32]],
    "static": false
}
```

## Post Training Quantization (PTQ)

`OpenVINOQuantization` pass and `OpenVINOQuantizationWithAccuracy` passes will run [Post-training quantization](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training.html) for OpenVINO models, as well as ONNX models, and support the uniform integer quantization method.
This method allows moving from floating-point precision to integer precision (for example, 8-bit) for weights and activations during the
inference time. It helps to reduce the model size, memory footprint and latency, as well as improve the computational efficiency, using
integer arithmetic. During the quantization process the model undergoes the transformation process when additional operations, that contain
quantization information, are inserted into the model. The actual transition to integer arithmetic happens at model inference.

Please refer to [OpenVINOQuantization](https://microsoft.github.io/Olive/reference/pass.html#openvinoquantization) for more details about the `OpenVINOQuantization` pass and its config parameters.

Please refer to [OpenVINOQuantizationWithAccuracy](https://microsoft.github.io/Olive/reference/pass.html#openvinoquantizationwithaccuracy) for more details about the `OpenVINOQuantizationWithAccuracy` pass and its config parameters.

### Example PTQ Configuration

```json
{
    "type": "OpenVINOQuantizationWithAccuracy",
    "data_config": "calib_data_config",
    "validation_func": "validate",
    "max_drop": 0.01,
    "drop_type": "ABSOLUTE"
}
```

## Weight Compression

`OpenVINOWeightCompression` pass runs [Weight Compression](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html) to compress Huggingface to OpenVINO model and Huggingface to ONNX model, as well as ONNX to ONNX model using Intel® NNCF.

Please refer to [OpenVINOWeightCompression](https://microsoft.github.io/Olive/reference/pass.html#openvinoweightcompression) for more details about the `OpenVINOWeightCompression` pass and its config parameters.

### Example Weight Compression Configuration

```json
{
    "type": "OpenVINOWeightCompression",
    "data_config": "compress_data_config",
    "transform_fn": "custom_transform_func",
    "extra_args": { "tokenizer": true },
    "compress_config": {
        "mode": "INT4_SYM",
        "ratio": 0.8
    }
}
```

## Model Encapsulation

`OpenVINOEncapsulation` pass is used to generate an onnx model that encapsulates a OpenVINO IR model. It supports `OpenVINOModelHandler` for now.

Please refer to [OpenVINOEncapsulation](https://microsoft.github.io/Olive/reference/pass.html#openvinoencapsulation) for more details about the pass and its config parameters.

### Example Encapsulation Configuration

```json
{
    "type": "OpenVINOEncapsulation",
    "target_device": "npu",
    "ov_version": "2025.1"
}
```

## Optimum CLI Command for Generative AI workloads

`OpenVINOOptimumConversion` pass will run [optimum-cli export openvino](https://huggingface.co/docs/optimum/main/en/intel/openvino/export) command on the input Huggingface models to convert those to OpenVINO models and perform weight compression and quantization if necessary to produce an output OpenVINO model.

Please refer to [OpenVINOOptimumConversion](https://microsoft.github.io/Olive/reference/pass.html#openvinooptimumconversion) and also to [optimum-cli export openvino](https://huggingface.co/docs/optimum/main/en/intel/openvino/export) for more details about the pass and its config parameters.

### Example Optimum Conversion Configuration

```json
{
    "type": "OpenVINOOptimumConversion",
    "extra_args" : { "device": "npu" },
    "ov_quant_config": {
        "weight_format": "int4",
        "dataset": "wikitext2",
        "awq": true
    }
}
```
