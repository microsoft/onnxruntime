---
title: Quantize ONNX Models
parent: Performance
nav_order: 3
redirect_from: /docs/how-to/quantization
---
# Quantize ONNX Models
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Quantization Overview
Quantization in ONNX Runtime refers to 8 bit linear quantization of an ONNX model.

 During quantization the floating point real values are mapped to an 8 bit quantization space and it is of the form:
 VAL_fp32 = Scale * (VAL_quantized - Zero_point)

 Scale is a positive real number used to map the floating point numbers to a quantization space. It is calculated as follows:

 For asymmetric quantization:
 ```
 scale = (data_range_max - data_range_min) / (quantization_range_max - quantization_range_min)
 ```

 For symmetric quantization:
 ```
 scale = abs(data_range_max, data_range_min) * 2 / (quantization_range_max - quantization_range_min)
 ```

 Zero_point represents zero in the quantization space. It is important that the floating point zero value be exactly representable in quantization space. This is because zero padding is used in many CNNs. If it is not possible to represent 0 uniquely after quantization, it will result in accuracy errors.

## ONNX quantization representation format
There are 2 ways to represent quantized ONNX models:
- Operator Oriented. All the quantized operators have their own ONNX definitions, like QLinearConv, MatMulInteger and etc.
- Tensor Oriented, aka Quantize and DeQuantize (QDQ). This format uses DQ(Q(tensor)) to simulate the quantize and dequantize process, and QuantizeLinear and DeQuantizeLinear operators also carry the quantization parameters. Models generated like below are in QDQ format:
  - Models quantized by quantize_static API below with quant_format=QuantFormat.QDQ.
  - Quantization-Aware training (QAT) models converted from Tensorflow or exported from PyTorch.
  - Quantized models converted from tflite and other framework.

For the latter 2 cases, you don't need to quantize the model with quantization tool. OnnxRuntime can run them directly as quantized model.

Picture below shows the equivalent representation with QDQ format and Operator oriented format for quantized Conv. This [E2E](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu/run.py) example demonstrates QDQ and Operator Oriented format.

![Changes to nodes from basic and extended optimizations](../../images/QDQ_Format.png)

## Quantizing an ONNX model
There are 3 ways of quantizing a model: dynamic, static and quantize-aware training quantization.

* **Dynamic quantization**: This method calculates the quantization parameter (scale and zero point) for activations dynamically.

* **Static quantization**: It leverages the calibration data to calculates the quantization parameter of activations.

* **Quantize-Aware training quantization**: The quantization parameter of activation are calculated while training, and the training process can control activation to a certain range.

### Quantization API
{: .no_toc}

Quantization has 3 main APIs, which corresponds to the 3 quantization methods:
* quantize_dynamic: dynamic quantization
* quantize_static: static quantization
* quantize_qat: quantize-aware training quantization

Please refer to [quantize.py](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/quantize.py) for quantization options for each method.

#### Example
{: .no_toc }

- Dynamic quantization

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'path/to/the/model.onnx'
model_quant = 'path/to/the/model.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant)
```

- Static quantization

  Please refer to [E2E_example_model](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization) for an example of static quantization.

### Method selection
{: .no_toc}

The main difference between dynamic quantization and static quantization is how scale and zero point of activation are calculated. For static quantization, they are calculated offline with calibration data set. All the activations have same scale and zero point. While for dynamic quantization, they are calculated on flight and will be specific for each activation, thus they are more accurate but introduce extra computation overhead.

In general, it is recommended to use dynamic quantization for RNN and transformer-based models, and static quantization for CNN models.

If both post-training quantization can not meet your accuracy goal, you can try quantization-aware training (QAT) to retrain the model. ONNX Runtime does not provide retraining at this time, but you can retrain your models with the original framework and reconvert them back to ONNX.

### Data type selection
{: .no_toc}

Quantization represents value with 8 bit, which can be either int8 and uint8. Combining with activation and weight, the data format can be (activation:uint8, weight:uint8), (activation:uint8, weight:int8), etc.

Let's use U8U8 as as shorthand for (activation:uint8, weight:uint8), and U8S8 for (activation:uint8, weight:int8), and S8U8, S8S8 for other two formats.

OnnxRuntime Quantization on CPU can run U8U8, U8S8 and S8S8. S8S8 with QDQ format is the default setting for blance of performance and accuracy. It should be the first choice. Only in cases that the accuracy drops a lot, you can try U8U8. Note that S8S8 with QOperator format will be slow on x86-64 CPUs and it should be avoided in general.
OnnxRuntime Quantization on GPU only support S8S8 format.

#### When and why do I need to try U8U8?
{: .no_toc }

On x86-64 machines with AVX2 and AVX512 extensions, OnnxRuntime uses VPMADDUBSW instruction for U8S8 for performance, but this instruction suffer saturation issue. Generally, it is not a big issue for final result. If you hit a big accuracy drop for some models, it may be caused by the saturation. In this case, you can either try [reduce_range](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/quantize.py) or U8U8 format which doesn't have saturation issue.

There is no such issue on other CPU archs(x64 with VNNI and ARM).

### List of Supported Quantized Ops
{: .no_toc}

Please refer to [registry](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/registry.py) for the list of supported Ops.

### Quantization and model opset versions
{: .no_toc}

Models must be opset10 or higher to be quantized. Models with opset < 10 must be reconverted to ONNX from its original framework using a later opset.

## Transformer-based models
There are specific optimization for transformer-based models, like QAttention for quantization of attention layer. In order to leverage those specific optimization, you need to optimize your models with [Transformer Model Optimization Tool](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers) before quantizing the model.

This [notebook](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/notebooks/bert) demonstrates the E2E process.

## Quantization on GPU

Hardware support is required to achieve better performance with quantization on GPUs. You need a device that support Tensor Core int8 computation, like T4, A100. Older hardware won't get benefit.

ORT leverage TRT EP for quantization on GPU now. Different with CPU EP, TRT takes in full precision model and calibration result for inputs. It decides how to quantize with their own logic. The overall procedure to leverage TRT EP quantization is:
- Implement a [CalibrationDataReader](https://github.com/microsoft/onnxruntime/blob/07788e082ef2c78c3f4e72f49e7e7c3db6f09cb0/onnxruntime/python/tools/quantization/calibrate.py).
- Compute quantization parameter with calibration data set. Our quantization tool supports 2 calibration methods: MinMax and Entropy. Note: In order to include all tensors from the model for better calibration, please run symbolic_shape_infer.py first. Please refer to[here](../execution-providers/TensorRT-ExecutionProvider.md#samples) for detail.
- Save quantization parameter into a flatbuffer file
- Load model and quantization parameter file and run with TRT EP.

We have 2 E2E examples [Yolo V3](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/object_detection/trt/yolov3) and [resnet50](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/trt/resnet50) for your reference.

## FAQ
### Why am I not seeing performance improvements?
{: .no_toc }

Performance improvement depends on your model and hardware. Quantization performance gain comes in 2 part: instruction and cache. Old hardware doesn't have or has few instruction support for byte computation. And quantization has overhead (quantize and dequantize), so it is not rare to get worse performance on old devices.

x86-64 with VNNI, GPU with Tensor Core int8 support and ARM with dot-product instructions can get better performance in general.

### Which method should I choose?
{: .no_toc}

Please refer to [here](#method-selection).

### When to use per-channel and reduce-range?
{: .no_toc}

Reduce-range will quantize the weight with 7-bits. It is designed for U8S8 format on AVX2 and AVX512 (non VNNI) machines to mitigate the [saturation issue](#data-type-selection). Don't need it on VNNI machine.

Per-channel quantization can improve the accuracy for models whose weight ranges are large. Try it firstly if the accuracy loss is large. And you need to enable reduce-range generally on AVX2 and AVX512 machines if per-channel is enabled.

### Why operators like MaxPool is not quantized?
{: .no_toc}

8-bit type support for some operators like MaxPool is added in ONNX opeset 12. Please check your model version and upgrade it to opset 12 and above if your model version is older.

