# Quantization tool Overview
This tool supports quantization of an onnx model. quantize() takes a model in ModelProto format and returns the quantized model in ModelProto format.

## Calibrating an ONNX model

Calibration can be used to improve quantization, adding reduced-precision computation for neural networks while retaining high accuracy without retraining.  Example usage:
```
python calibrate.py --model_path=<'path/to/model.onnx'> --dataset_path=<'path/to/data/folder'> --calib_mode='naive'
```  
Note the option `--dataset_path`, used to specify the path for the "calibration dataset".  This dataset is meant to contain "representative" examples of the input (presumably, a subset of the dataset the original model was trained on), which are used to collect statistics on the "typical" outputs of selected nodes in the model, as described in the sections below.  The dataset can be provided as:

* A filepath for a directory, containing images or
* A protobuf file, encoding a set of images using the `TensorProto` schema,

Whether a set of files or a collected bundles of tensors, `calibrate.py` assumes that the images in the representative dataset are suitable for immediate consumption by the original fp32 model, i.e., that they have been adequately pre-processed.  `calibrate.py` also supports a small set of preprocessing functions, selectable via the CLI option `--data_preprocess`.  Currently, this option accepts the values:

* 'preprocess_method1' : reshapes an image into NCHW format and scales the pixel values to the[-1, 1] range.  This method mirrors [the treatment for mobilenet models available in version 0.5 of mlperf](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py#L226)
* 'preprocess_method2' : resizes and normalizes image to NCHW format, in a [technique used by mlperf 0.5](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py#L250) for variants of ResNet.

For maximum flexibility, it is recommended that the user carries out the necessary preprocessing as a separate stage in the quantization pipeline, and provides the already-preprocessed dataset to `calibrate.py`.  Alternatively, we welcome contributions of preprocessing techniques (see below).

### How does this work

`calibrate.py` adds `ReduceMin` and `ReduceMax` nodes to all `Conv` and `MatMul` nodes in a loaded ONNX model and ensures their outputs are stored as part of the graph output, extracts intermediate output values after inference, and returns a dictionary mapping added node names to average (`ReduceMin`, `ReduceMax`) values as input to `quantize.py`.

### How to add preprocessing options

To add the current set of available preprocessing functions, you need to add a Python function to the `data_preprocess.py` file.  Such a function needs to take a filename and the (width, height) of an image as integers, and return a `numpy` `ndarray`.

To expose it a new preprocessing function to the command line, you'll need to "register" it by adding it to the name/function mapping maintained as a dictionary in the `set_preprocess` function in `data_preprocess.py`.

### End-to-end example

Say you want to quantize a model `/models/mymodel.onnx` that has been trained with a subset of the ILSVRC ImageNet dataset.  In what follows it is assumed that you have either the dataset or a sample you deem significant to use as a calibration set --  one way to get a proxy for ImageNet is to follow a "fake" imageset as [this notebook from MLPerf](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/GettingStarted.ipynb) provides.  Assume that the images from the dataset are in the `/tmp/data/ilsvrc12` directory (note you don't need the labels to quantize, but they are useful to validate/verify the results of the quantization).  You can then run

```
python calibrate.py \
   --model_path=/models/mymodel.onnx \
   --dataset_path=/tmp/data/ilsvrc12 \
   --calib_mode='naive' \
   --data_preprocess=preprocess_method1
```

which specifies that images are to be preprocessed before they're fed to the model for the inference step (see above for description of `preprocess_method1`) necessary to calculate the quantization parameters used later in the actual quantization process.

If your model requires a preprocessing of the input that is not provided with the scripts, you'll need to resort to preprocessing "offline", i.e., carry out the following process
1. read each image, running it through whatever preprocess is required for the model, 
1. store the resulting tensor, and
1. "stack" the tensors corresponding to all the images in the calibration dataset in a tensor, serializing it to the `onnx` `TensorProto` schema (refer, for example to [this article](https://onnx.ai/onnx-r/articles/protobufs.html), with code in R, or to [the manual](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md) for code in Python).

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
