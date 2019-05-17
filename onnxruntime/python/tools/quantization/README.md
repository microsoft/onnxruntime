# Quantization tool Overview
This tool supports quantization of an onnx model. quantize() takes a model in ModelProto format and returns the quantized model in ModelProto format.

## Quantize an onnx model
```python
import onnx
from quantize import quantize, QuantizationMode

# Load the onnx model
model = onnx.load('path/to/the/model.onnx')
# Quantize
quantized_model = quantize(model, per_channel=False, quantization_mode=QuantizationMode.IntegerOps_Dynamic)
# Save the quantized model
onnx.save(quantized_model, 'path/to/the/quantized_model.onnx')
```

## Quantization modes

The following quantization modes are supported.
- **QuantizationMode.IntegerOps_Static**:
    Quantize using integer ops. Only [ConvInteger](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvInteger) and [MatMulInteger](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMulInteger) ops are supported now.
    Inputs/activations are quantized using static scale and zero point values.
    These values are specified through "input_quantization_params" option.

- **QuantizationMode.IntegerOps_Dynamic**:
    Quantize using integer ops. Only ConvInteger and MatMulInteger ops are supported now.
    Inputs/activations are quantized using dynamic scale and zero point values.
    These values are computed while running the model.

- **QuantizationMode.QLinearOps_Static**:
    Quantize using QLinear ops. Only [QLinearConv](https://github.com/onnx/onnx/blob/master/docs/Operators.md#qlinearconv) and [QLinearMatMul](https://github.com/onnx/onnx/blob/master/docs/Operators.md#QLinearMatMul) ops are supported now.
    Inputs/activations are quantized using static scale and zero point values.
    These values are specified through "input_quantization_params" option.
    In this mode, output scale and zero point values need to be specified using "output_quantization_params" option.

- **QuantizationMode.QLinearOps_Dynamic**:
    Quantize using QLinear ops. Only QLinearConv and QLinearMatMul ops are supported now.
    Inputs/activations are quantized using dynamic scale and zero point values.
    These values are computed while running the model.
    In this mode, output scale and zero point values need to be specified using "output_quantization_params" option.

## Options

See below for a description of all the options to quantize():

- **model**: ModelProto to quantize
- **per_channel**: *default: True*
	If True, weights of Conv nodes are quantized per output channel. 
	If False, they are quantized per tensor. Refer [QLinearConv](https://github.com/onnx/onnx/blob/master/docs/Operators.md#qlinearconv) for more information.
- **nbits**: *default: 8*
	Number of bits to represent quantized data. Currently only nbits=8 is supported.
- **quantization_mode**: *default: QuantizationMode.IntegerOps_Dynamic*
	Can be one of the QuantizationMode values.
- **asymmetric_input_types**: *default: False*
	If True, weights are quantized into signed integers and inputs/activations into unsigned integers.
	If False, weights and inputs/activations are quantized into unsigned integers.
- **input_quantization_params**: *default: None*
	Dictionary to specify the zero point and scale values for inputs to conv and matmul nodes.
        Used in QuantizationMode.IntegerOps_Static or QuantizationMode.QLinearOps_Static mode.
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
        Used in QuantizationMode.QLinearOps_Static or QuantizationMode.QLinearOps_Dynamic mode.
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