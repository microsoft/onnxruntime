# Quantization and Calibration Tools

Quantization in ORT refers to 8 bit linear quantization of an onnx model. There are 2 tools which aid converting an onnx model to an onnx quantized model.

    * Quantization Tool
    * Calibration Tool

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
There are 2 ways of quantizing a model

* Only use quantization : This method assumes the model owner is going to use Integer Ops for quantization or has pre calculated the quantization params as they are required inputs for using QLinear Ops

ONNX Model ---> quantize.py ---> ONNX Quantized Model

* Use both calibration and quantization : This method is preferred when using QLinear Ops for quantization.

ONNX Mode --> calibrate.py --> quantize.py --> ONNX Quantized model

Today ORT does not guarantee support for E2E model quantization, meaning since not all ONNX ops have support for 8 bit data types therefore only the supported ops in the model are quantized. For rest of the ops inputs are reconverted to FP32.

### List of Supported Quantized Ops:
The following ops were chosen as phase 1 ops because in most of the CNN models these ops consume most amount of compute and power and therefore there is benefit in quantizing these ops to get perf benefits.
 * Convolution
 * Matmul
 * Data type agnostic ops like transpose, identity etc. ( Note: special quantization is not done for these ops.)

 ### Quantization and model opset versions
Quantization is fairly new in ONNX and ONNXRuntime. Quantization ops were introduced in ONNX opset version 10. Therefore it is important that the model which is being quantized be opset 10 or higher. In case the model opset version is < 10 then it is recommended that the model should be reconverted to ONNX from its original framework using the latest opset.

Quantization tool displays a warning when the model opset version is < 10 and still goes ahead and quantizes the model and at the end changes the opset version to 10. It is the responsibility of the model owner to run model checker and make sure the model is valid. If the model is not valid then use the above recommended way i.e. reconvert the model from original framework.

### Quantization and Graph Optimization
Please note quantization and graph optimizations may not always work together.

#### Quantizing an optimized model
If a model is optimized using level 99 (i.e. all possible optimizations are run on that model) then it is possible that after these optimizations are applied the model is converted in a way that quantization cannot be applied on this model anymore and therefore after running quantization script there will be no change in the model.

#### Optimizing a quantized model
Same goes the other way round. After quantizing a model some graph optimizations which otherwise might have been applicable on this model may not be applicable anymore.

It is advised that the model owner be aware of this and run perf evaluations to understand which technique gives the best performance for their model.

## Quantization tool
quantize() takes a model in ModelProto format and returns the quantized model in ModelProto format.

### Various quantization modes
Default is set to QuantizationMode.IntegerOps with dynamic input quantization.

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
    Quantize using integer ops. Inputs/activations are quantized using dynamic scale and zero point values which are computed while running the model. This is the default quantization mode.
    ```python
    quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps, static=False)
    ```

- **QuantizationMode.QLinearOps with static input quantization**:
    Quantize using QLinear ops. Inputs/activations are quantized using static scale and zero point values which are specified through "quantization_params" option.
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

### Options

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
        example:
        [
            'Conv__224',
            'Conv__252'
        ]

### Example - Quantize an ONNX Model
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

## Calibration tool
Calibration can be used to improve quantization, adding reduced-precision computation for neural networks while retaining high accuracy without retraining.

Calibration uses a small data set representative of the original data set to calculate quantization thresholds. To calculate the quantization thresholds it updates the original onnx model by adding `ReduceMin` and `ReduceMax` nodes to all the nodes which are candidates for quantization (Today this is applicable for `Conv` and `MatMul` nodes). It then runs through the calibration datasets to gather these outputs and finally calculates the quantization thresholds. These are then passed as inputs to quantize.py for quantizing the model.

### Options

See below for a description of all the options to calibrate():

- **model_path**: Path to the original FP32 model
- **data_reader**: User-implemented object to read in and preprocess calibration dataset based on CalibrationDataReader interface, which takes in  `calibration_image_data` and can generate the next input data dictionary for ONNXinferencesession run.
- **op_types**: Operator types to be calibrated and quantized, *default = 'Conv,MatMul'*
- **black_nodes**: Operator names that should not be calibrated and quantized, *default = ''*
- **white_nodes**: Operator names that force to be calibrated and quantized, *default = ''*
- **augmented_model_path**: Path to save the augmented_model.


### End-to-end example
This is an E2E example to demonstrate calibration, quantization and accuracy testing for a ResNet50 model. As discussed above, if you want to use the quantization tool only, please follow the example above in `Quantization Tool` section.

We leverage the instructions as the following:

* Download the model : Download the [resnet50_v1](./E2E_example_model/resnet50_v1.onnx).

* Install latest versions of ONNX and ONNXRuntime.

* Download the test calibration data set: 
    * A `calibration_data_set_test` folder is included under `./E2E_example_model`. It is used as the test calibration data set for this E2E example.

* Run the E2E example. [e2e_example](./E2E_example_model/e2e_user_example.py).
    * `ResNet50DataReader`is implemented based on `CalibrationDataReader` interface and it's used specifically for reading in the image data for ResNet50.`preprocess_func` is used by `ResNet50DataReader`to load and preprocess the image data.
        - *preprocess_func*: resizes and normalizes image to NHWC format, in a [technique used by mlperf 0.5](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py#L250) for variants of ResNet.
        - Alternatively, if user wants to accept preprocessed tensors in .pb format. Refer to [this article](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#manipulating-tensorproto-and-numpy-array) to understand how to hop between numpy arrays and tensorproto and write corresponding preprocess function.
    * Run the calibration tool:
    ```
    python3 e2e_user_example.py
    ```
    * After successfuly running the E2E example, a `calibrated_quantized_model` will be saved. (The `quantization_mode` used here is QLinear Ops.)

* Setup and run mlperf accuracy tests : Now that quantized model is ready run the accuracy tests using the mlperf accuracy benchmarks.
    * Set up the [mlperf benchmark](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection#prerequisites-and-installation)
    * Run accuracy test : For example
    ```
    ./run_local.sh  onnxruntime resnet50 --accuracy --count 5000
    ```
