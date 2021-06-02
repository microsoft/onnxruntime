---
title: RKNPU
parent: Execution Providers
grand_parent: Reference
---

# RKNPU Execution Provider
*PREVIEW*

RKNPU DDK is an advanced interface to access Rockchip NPU. The RKNPU Execution Provider enables deep learning inference on Rockchip NPU via RKNPU DDK.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Build 
For build instructions, please see the [BUILD page](../../how-to/build/eps.md#rknpu).

## Usage
**C/C++**

To use RKNPU as an execution provider for inferencing, please register it as below.
```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_RKNPU(sf));
Ort::Session session(env, model_path, sf);
```
The C API details are [here](../api/c-api.md).


## Support Coverage
### Supported Platform
{: .no_toc }

* RK1808 Linux

*Note: RK3399Pro platform is not supported.*
### Supported Operators
{: .no_toc }


The table below shows the ONNX Ops supported using the RKNPU Execution Provider and the mapping between ONNX Ops and RKNPU Ops.

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


### Supported Models
{: .no_toc }


The following models from the ONNX model zoo are supported using the RKNPU Execution Provider

**Image Classification**
- squeezenet
- mobilenetv2-1.0
- resnet50v1
- resnet50v2
- inception_v2

**Object Detection**
- ssd
- yolov3