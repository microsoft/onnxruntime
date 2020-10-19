# RKNPU Execution Provider (preview)
RKNPU DDK is an advanced interface to access Rockchip NPU. RKNPU Execution Provider enables deep learning inference on Rockchip NPU via RKNPU DDK.

## Supported platforms

* RK1808 Linux

*Note: RK3399Pro platform is not supported.*


## Build 
For build instructions, please see the [BUILD page](../../BUILD.md#RKNPU).

## Usage
### C/C++
To use RKNPU as execution provider for inferencing, please register it as below.
```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_RKNPU(sf));
Ort::Session session(env, model_path, sf);
```
The C API details are [here](../C_API.md#c-api).


## Supported Operators

The table below shows the ONNX Ops supported using RKNPU Execution Provider and the mapping between ONNX Ops and RKNPU Ops.

| **ONNX Ops** | **RKNPU Ops** |
| --- | --- |
| Add | ADD |
| Mul | MULTIPLY |
| Conv | CONV2D |
| QLinearConv | CONV2D |
| Gemm | FULLCONNECT |
| Softmax | SOFTMAX |
| AveragePool | POOL |
| GlobalAveragePool | POOL |
| MaxPool | POOL |
| GlobalMaxPool | POOL |
| LeakyRelu | LEAKY_RELU |
| Concat | CONCAT |
| BatchNormalization | BATCH_NORM |
| Reshape | RESHAPE |
| Flatten | RESHAPE |
| Squeeze | RESHAPE |
| Unsqueeze | RESHAPE |
| Transpose | PERMUTE |
| Relu | RELU |
| Sub | SUBTRACT |
| Clip(0~6)| RELU6 |
| DequantizeLinear | DATACONVERT |
| Clip | CLIP |


## Supported Models

Below Models are supported from ONNX open model zoo using RKNPU Execution Provider

### Image Classification
- squeezenet
- mobilenetv2-1.0
- resnet50v1
- resnet50v2
- inception_v2

### Object Detection
- ssd
- yolov3