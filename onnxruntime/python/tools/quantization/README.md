# Quantization Tools

Quantization in ORT refers to 8 bit linear quantization of an onnx model.

## Quantization specifics
 During quantization the floating point real values are mapped to an 8 bit quantization space and it is of the form :
 VAL_fp32 = Scale * (VAL_quantized - Zero_point)

 Scale is a positive real number used to map the floating point numbers to a quantization space. It is calculated as follows :
 For unsigned 8 bit
 ```
 scale = (data_range_max - data_range_min) / (quantization_range_max - quantization_range_min)
 ```

 For signed 8 bit
 ```
 scale = Abs(data_range_max, data_range_min) * 2 / (quantization_range_max - quantization_range_min)
 ```

 Zero point represents zero in quantization space. It is important that floating point zero value be exactly representable in quantization space. This is because in lot of CNNs, zero padding is used and if after quantization it is not possible to represent 0 uniquely then it will lead to accuracy errors.

## Quantizing an onnx model
There are 3 ways of quantizing a model: dynamic, static and auantize-aware training quantization.

* Dynamic quantization : This method calculates the quantization parameter (scale and zero point) for activations dynamically.

* Static quantization: It leverages the calibration data to calculates the quantization parameter of activations.

* Quantize-Aware training quantization. The quantization parameter of activation are calculated while training, and the training process can control activation to a certain range.

### List of Supported Quantized Ops:
The following ops were chosen as phase 1 ops because in most of the CNN models these ops consume most amount of compute and power and therefore there is benefit in quantizing these ops to get perf benefits.
 * Conv
 * MatMul
 * MaxPool
 * Relu
 * Clip
 * Add (Experimental)
 * Mul (Experimental)

 ### Quantization and model opset versions
Quantization is fairly new in ONNX and ONNXRuntime. Quantization ops were introduced in ONNX opset version 10. Therefore it is important that the model which is being quantized be opset 10 or higher. In case the model opset version is < 10 then it is recommended that the model should be reconverted to ONNX from its original framework using the latest opset.

### Quantization and Graph Optimization
Please note quantization and graph optimizations may not always work together.

#### Quantizing an optimized model
If a model is optimized using level 99 (i.e. all possible optimizations are run on that model) then it is possible that after these optimizations are applied the model is converted in a way that quantization cannot be applied on this model anymore and therefore after running quantization script there will be no change in the model.

#### Optimizing a quantized model
Same goes the other way round. After quantizing a model some graph optimizations which otherwise might have been applicable on this model may not be applicable anymore.

It is advised that the model owner be aware of this and run perf evaluations to understand which technique gives the best performance for their model.

## Quantization API
Quantization has 3 main APIs quantize_dynamic, quantize_static, and quantize_qat, which corresponds to dynamic quantization, static quantization and quantize-aware training quantization respectively.

### Options

See below for a description of the common options to quantize_dynamic, quantize_static and quantize_qat:

- **model_input**:
  - 
    file path of model to quantize
- **model_output**:
  - 
    file path of model to quantize
- **op_types_to_quantize**: *defalut: []
  - 
    specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default.
- **per_channel**: *default: False*
  - 
    If True, weights of Conv nodes are quantized per output channel.
  
    If False, they are quantized per tensor. Refer [QLinearConv](https://github.com/onnx/onnx/blob/master/docs/Operators.md#qlinearconv) for more information.
- **activation_type**: *defalut: QuantType.QUInt8*
  - 
    quantization data type of activation. It can be QuantType.QInt8 or QuantType.QUInt8
- **weight_type**: *defalut: QuantType.QUInt8*
  - 
    quantization data type of weight. It can be QuantType.QInt8 or QuantType.QUInt8
- **nodes_to_quantize**: *default: []*
  - 
    List of nodes names to quantize. When this list is not None only the nodes in this list
    are quantized.
    example:
    [
        'Conv__224',
        'Conv__252'
    ]
- **nodes_to_exclude**: *default: []*
  - 
    List of nodes names to exclude. The nodes in this list will be excluded from quantization
    when it is not None.

In addition, user needs to provide an implementation of CalibrationDataReader for quantize_static CalibrationDataReader takes in the calibration data and generates input of the model

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

Please refer to ./E2E_example_model for an example of static quantization.
