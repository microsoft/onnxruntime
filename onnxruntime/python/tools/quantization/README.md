# Quantization tool Overview
This tool supports quantization of an onnx model. quantize() takes a model in ModelProto format and returns the quantized model in ModelProto format.

## Quantize an onnx model
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
    Quantize using integer ops. Inputs/activations are quantized using static scale and zero point values which are specified through "input_quantization_params" option.
    ```python
    quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps,
                               static=True,
                               input_quantization_params={
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
                               input_quantization_params={
                                    'input_1': [np.uint8(113), np.float32(0.05)]
                               },
                               output_quantization_params={
                                    'output_1': [np.uint8(113), np.float32(0.05)]
                               })
    ```

- **QuantizationMode.QLinearOps with dynamic input quantization**:
    Quantize using QLinear ops. Inputs/activations are quantized using dynamic scale and zero point values which are computed while running the model.
    Output scale and zero point values have to be specified using "output_quantization_params" option.
    ```python
    quantized_model = quantize(model, quantization_mode=QuantizationMode.QLinearOps,
                               static=False,
                               output_quantization_params={
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
If True, the inputs/activations are quantized using static scale and zero point values specified through input_quantization_params.
If False, the inputs/activations are quantized using dynamic scale and zero point values computed while running the model.
- **asymmetric_input_types**: *default: False*
    If True, weights are quantized into signed integers and inputs/activations into unsigned integers.
    If False, weights and inputs/activations are quantized into unsigned integers.
- **input_quantization_params**: *default: None*
    Dictionary to specify the zero point and scale values for inputs to conv and matmul nodes.
        Should be specified when static is set to True.
        The input_quantization_params should be specified in the following format:
            {
                "input_name": [zero_point, scale]
            }.
        zero_point should be of type np.uint8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_1:0': [np.uint8(0), np.float32(0.019539741799235344)],
                'resnet_model/Relu_2:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }
- **output_quantization_params**: *default: None*
    Dictionary to specify the zero point and scale values for outputs of conv and matmul nodes.
    Should be specified in QuantizationMode.QLinearOps mode.
        The output_quantization_params should be specified in the following format:
            {
                "output_name": [zero_point, scale]
            }
        zero_point can be of type np.uint8/np.int8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_3:0': [np.int8(0), np.float32(0.011359662748873234)],
                'resnet_model/Relu_4:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }