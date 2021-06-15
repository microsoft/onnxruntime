---
title: Quantize ONNX Models
parent: How to
nav_order: 4
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

 For unsigned 8 bit
 ```
 scale = (data_range_max - data_range_min) / (quantization_range_max - quantization_range_min)
 ```

 For signed 8 bit
 ```
 scale = abs(data_range_max, data_range_min) * 2 / (quantization_range_max - quantization_range_min)
 ```

 Zero point represents zero in the quantization space. It is important that the floating point zero value be exactly representable in quantization space. This is because zero padding is used in many CNNs. If it is not possible to represent 0 uniquely after quantization, it will result in accuracy errors.

## Quantizing an ONNX model
There are 3 ways of quantizing a model: dynamic, static and quantize-aware training quantization.

* **Dynamic quantization**: This method calculates the quantization parameter (scale and zero point) for activations dynamically.

* **Static quantization**: It leverages the calibration data to calculates the quantization parameter of activations.

* **Quantize-Aware training quantization**: The quantization parameter of activation are calculated while training, and the training process can control activation to a certain range.

### Method selection
The main difference between dynamic quantization and static quantization is how scale and zero point of activation is calculated. For static quantization, they are calculated offline with calibration data set. All the activations have same scale and zero point. While for dynamic quantization, they are calculated on flight and will be specific for each activation, thus they are more accurate but introduce extra computation overhead.

In general, it is recommended to use dynamic quantization for RNN and transformer-based models, and static quantizaiton for CNN models.

If both post-training quantization can not meet your accuracy goal, you can try quantization-aware training to retrain the model. OnnxRuntime doesn't provide retrain capability now. You can retrain you model with original framework and then converted it back to ONNX. 

## List of Supported Quantized Ops
Please refer to [registry](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/registry.py) for the list of supported Ops.

## Quantization and model opset versions
 Quantization ops were introduced in ONNX opset version 10, so the model which is being quantized must be opset 10 or higher. If the model opset version is < 10 then the model should be reconverted to ONNX from its original framework using a later opset.

## Quantization API
Quantization has 3 main APIs, which corresponds to the 3 quantization method:
* quantize_dynamic: dynamic quantization
* quantize_static: static quantization
* quantize_qat: quantize-aware training quantization

Please refer to [quantize.py](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/quantize.py) for quantization options for each method.

### Example
- Dynamic quantization
```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'path/to/the/model.onnx'
model_quant = 'path/to/the/model.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
```

- QAT quantization
```python
import onnx
from onnxruntime.quantization import quantize_qat, QuantType

model_fp32 = 'path/to/the/model.onnx'
model_quant = 'path/to/the/model.quant.onnx'
quantized_model = quantize_qat(model_fp32, model_quant)
```

- Static quantization

  Please refer to [E2E_example_model](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/quantization/E2E_example_model) for an example of static quantization.

## Which quantization data type to use?
Quantization represents value with 8 bit, which can be either int8 and uint8. Combining with activation and weight, the data format can be (activation:uint8, weight:uint8), (activation:uint8, weight:int8), etc.

Let's use U8U8 as as shorthand for (activation:uint8, weight:uint8), and U8S8 for (activation:uint8, weight:int8), and S8U8, S8S8 for other two formats.

Currently, OnnxRuntime CPU only supports activation with type uint8, i.e., U8X8 only.

### x86-64
#### AVX2
Try U8U8 firstly, and then U8S8.
- Performance
U8S8 leverage VPMADDUBSW instruction but U8U8 kernel can process 6 rows at a time versus 4 rows for the U8S8 kernel, and U8U8 sequence is two instructions vs three instructions for U8S8. Thus, this balance ends up with u8u8 just in reach of U8S8 for older HW (Broadwell)
- Accuracy
VPMADDUBSW has saturation issue, thus U8S8 needs to use [reduce_range](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/quantize.py) if accuracy is not good enough.

#### AVX512
U8S8 is better in general
- Performance
U8S8 can only do twice mul/adds per instruction as much as U8U8. Typically U8S8 sequence ends up winning, can be twice as fast as u8u8.
- Accuracy No difference

#### VNNI
No difference.

### ARM64
U8S8 can be faster than U8U8 for low end ARM64 and no difference on accuracy. There is no difference for high end ARM64.

## Transformer-based models
We have specific optimization for transformer-based models, like QAttention for quantization of attention layer. In order to leverage those specific optimization, you need to optimize your models with [Transformer Model Optimization Tool](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers) before quantizing the model.

We have a [notebook](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/quantization/notebooks/bert) that demonstrates the E2E process.

## Quantization on GPU
Hardware suppor is required to achieve better performance with quantization on GPUs. You need a device that supports Tensor Core int8 computation, like T4, A100. Older hardware won't get benefit.

We leverage TRT EP for quantization on GPU. Different with CPU EP, TRT takes in full precision model and calibration result for inputs. It decides how to quantize with their own logic. The overall procedure to leverage TRT EP quantization is:
- Implement a [CalibrationDataReader](https://github.com/microsoft/onnxruntime/blob/07788e082ef2c78c3f4e72f49e7e7c3db6f09cb0/onnxruntime/python/tools/quantization/calibrate.py).
- Compute quantization parameter with calibration data set. Our quantization tool supports 2 calibration methods: MinMax and Entropy. Note: In order to include all tensors from the model for better calibration, please run symbolic_shape_infer.py first. Please refer to[here](https://www.onnxruntime.ai/docs/reference/execution-providers/TensorRT-ExecutionProvider.html#sample) for detail.
- Save quantization parameter into a flatbuffer file
- Load model and quantization parameter file and run with TRT EP.

We have 2 E2E examples [Yolo V3](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/quantization/E2E_example_model/object_detection/trt/yolov3) and [resnet50](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/quantization/E2E_example_model/image_classification/trt/resnet50) for your reference.

## FAQ
### Why performance is not better or even worse?
Performance improvement depends on your model and hardware. Quantization performance gain comes in 2 part: instruction and cache. Old hardware doesn't have or has few instrction support for byte computation. And quantization has overhead (quantize and dequantize), so it is not rare to get worse performance on old devices.

x86-64 with VNNI, GPU with Tensor Core int8 support and ARM with dot-product instructions can get better performance in general.

### Which method should I choose?
Please refer to [here](#method-selection).

### When to use per-channel and reduce-range?
Reduce-range will quantize the weight with 7-bits. It is designed for U8S8 format on AVX2 machines to mitigate the [saturation issue](#avx2). Don't use it if you are not using U8S8 on AVX2 machines.

Per-channel quantization can improve the accuracy for models whose weight ranges are large. Try it firstly if the accuracy loss is large.