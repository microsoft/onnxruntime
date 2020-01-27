# Quantization tool Overview
This tool supports 8 bit linear quantization of an onnx model. quantize() takes a model in ModelProto format and returns the quantized model in ModelProto format.
Today ORT does not guarantee support for E2E model quantization, meaning since not all ONNX ops have support for 8 bit data types therefore only the supported ops in the model are quantized. For rest of the ops inputs are reconverted to FP32.

List of Supported Quantized Ops:
The following ops were chosen as phase 1 ops because in most of the CNN models these ops consume most amount of compute and power and therefore there is benefit in quantizing these ops to get perf benefits.
 * Convolution
 * Matmul
 * Data type agnostic ops like transpose, identity etc ( Note: special quantization is not done for these ops. )

 ## Quantization specifics
 ONNX implements 8 bit linear quantization. During quantization the floating point real values are mapped to a 8 bit quantization space and it is of the form :
 VAL_fp32 = Scale * (VAL_quantized - Zero_point)
 
 Scale is a positive real number used to map the floating point numbers to a quantization space. It is calculated as follows : 
 For unsigned 8 bit
 ```
 scale = (data_rage_max - data_range_min) / (quantization_range_max - quantization_range_min)
 ```

 For signed 8 bit
 ```
 scale = Abs(data_rage_max, data_range_min) * 2 / (quantization_range_max - quantization_range_min)
 ```

 Zero point represents zero in quantization space. It is important that floating point zero value be exactly representable in quantization space. This is because in lot of CNNs, zero padding is used and if after quantization it is not possible to represent 0 uniquely then it will lead to accuracy errors.

## Quantization and model opset versions
Quantization is fairly new in ONNX and ONNXRuntime. Quantization ops were introduced in ONNX opset version 10. Therefore it is important that the model which is being quantized be opset 10 or higher. In case the model opset version is < 10 then it is recommended that the model should be reconverted to ONNX from its original framework using the latest opset.

Quantization tool displays a warning when the model opset version is < 10 and still goes ahead and quantizes the model and at the end changes the opset version to 10. It is the responsibility of the model owner to run model checker and make sure the model is valid. If the model is not valid then use the above recommended way i.e. reconvert the model from original framework.

## Quantization and Graph Optimization
Please note quantization and graph optimizations may not always work together.

### Quantizing an optimized model
If a model is optimized using level 99 (i.e. all possible optimizations are run on that model) then it is possible that after these optimizations are applied the model is converted in a way that quantization cannot be applied on this model anymore and therefore after running quantization script there will be no change in the model. 

### Optimizing a quantized model
Same goes other way round. After quantizing a model some graph optimizations which otherwise might have been applicable on this model may not be applicable anymore. 

It is advised that the model owner be aware of this and run perf evaluations to understand which technique gives the best performance for their model.

## Quantize an ONNX model
```python
import onnx
from quantize import quantize, QuantizationMode

# Load the onnx model
model = onnx.load('path/to/the/model.onnx')
# Quantize
quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps)
# Save the quantized model
onnx.save(quantized_model, 'path/to/the/quantized_model.onnx')
```

## Examples of various quantization modes

- **QuantizationMode.IntegerOps with static input quantization**:
    Quantize using integer ops. Inputs/activations are quantized using static scale and zero point values which are specified through "quantization_params" option.
    ```python
    quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps,
                               static=True,
                               quantization_params={
                                    'input_1': [np.uint8(113), np.float32(0.05)]
                               })
    ```

- **QuantizationMode.IntegerOps with dynamic input quantization**:
    Quantize using integer ops. Inputs/activations are quantized using dynamic scale and zero point values which are computed while running the model.
    ```python
    quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps, static=False)
    ```

- **QuantizationMode.QLinearOps with static input quantization**:
    Quantize using QLinear ops. Inputs/activations are quantized using static scale and zero point values which are specified through "input_quantization_params" option.
    Output scale and zero point values have to be specified using "output_quantization_params" option.
    ```python
    quantized_model = quantize(model, quantization_mode=QuantizationMode.QLinearOps,
                               static=True,
                               quantization_params={
                                    'input_1': [np.uint8(113), np.float32(0.05)]
                                    'output_1': [np.uint8(113), np.float32(0.05)]
                               })
    ```

- **QuantizationMode.QLinearOps with dynamic input quantization**:
    Quantize using QLinear ops. Inputs/activations are quantized using dynamic scale and zero point values which are computed while running the model.
    Output scale and zero point values have to be specified using "quantization_params" option.
    ```python
    quantized_model = quantize(model, quantization_mode=QuantizationMode.QLinearOps,
                               static=False,
                               quantization_params={
                                    'output_1': [np.uint8(113), np.float32(0.05)]
                               })
    ```

## Options

See below for a description of all the options to quantize():

- **model**: ModelProto to quantize
- **per_channel**: *default: False*
    If True, weights of Conv nodes are quantized per output channel.
    If False, they are quantized per tensor. Refer [QLinearConv](https://github.com/onnx/onnx/blob/master/docs/Operators.md#qlinearconv) for more information.
- **nbits**: *default: 8*
    Number of bits to represent quantized data. Currently only nbits=8 is supported.
- **quantization_mode**: *default: QuantizationMode.IntegerOps*
*QuantizationMode.IntegerOps*:  Quantize using integer ops. Only [ConvInteger](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvInteger) and [MatMulInteger](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMulInteger) ops are supported now.
*QuantizationMode.QLinearOps*: Quantize using QLinear ops. Only [QLinearConv](https://github.com/onnx/onnx/blob/master/docs/Operators.md#qlinearconv) and [QLinearMatMul](https://github.com/onnx/onnx/blob/master/docs/Operators.md#QLinearMatMul) ops are supported now.
- **static**: *default:False*
If True, the inputs/activations are quantized using static scale and zero point values specified through quantization_params.
If False, the inputs/activations are quantized using dynamic scale and zero point values computed while running the model.
- **asymmetric_input_types**: *default: False*
    If True, weights are quantized into signed integers and inputs/activations into unsigned integers.
    If False, weights and inputs/activations are quantized into unsigned integers.
- **force_fusions**: *default: False*
    If True, nodes added for dynamic quantization are fused.
    If False, no fusion is applied for nodes which are added for dynamic quantization.
    This optimization is available from opset 11.
- **quantization_params**: *default: None*
    Dictionary to specify the zero point and scale values for inputs to and outputs from conv and matmul nodes.
        Should be specified when static is set to True.
        The quantization_params should be specified in the following format:
            {
                "input_name": [zero_point, scale]
            }.
        zero_point should be of type np.uint8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_1:0': [np.uint8(0), np.float32(0.019539741799235344)],
                'resnet_model/Relu_2:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }
- **nodes_to quantize**: *default: None*
    List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        exmaple:
        [
            'Cov__224',
            'Conv__252'
        ]
