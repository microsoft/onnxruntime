---
title: Getting Started - TensorFlow
nav_exclude: true 
parent: Accelerate TensorFlow
grand_parent: Inferencing
---

# Getting Started: Converting TensorFlow to ONNX

TensorFlow models (including keras and TFLite models) can be converted to ONNX using the [tf2onnx](https://github.com/onnx/tensorflow-onnx) tool.

## Installation

First install tf2onnx in a python environment that already has TensorFlow installed.

`pip install tf2onnx` (stable)

**OR**

`pip install git+https://github.com/onnx/tensorflow-onnx` (latest from GitHub)

## Converting a Model

### Keras models and tf functions

Keras models and tf functions and can be converted directly within python:

```python
import tensorflow as tf
import tf2onnx
import onnx

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, activation="relu"))

input_signature = [tf.TensorSpec([3, 3], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, "dst/path/model.onnx")
```

See the [Python API Reference](https://github.com/onnx/tensorflow-onnx#python-api-reference) for full documentation.

### SavedModel

Convert a TensorFlow saved model with the command:

`python -m tf2onnx.convert --saved-model path/to/savedmodel --output dst/path/model.onnx --opset 13`

`path/to/savedmodel` should be the **path to the directory containing** `saved_model.pb`

See the [CLI Reference](https://github.com/onnx/tensorflow-onnx#cli-reference) for full documentation.

### TFLite

tf2onnx has support for converting tflite models. Add the optional `--dequantize` flag to remove quantization.

`python -m tf2onnx.convert --tflite path/to/model.tflite --output dst/path/model.onnx --opset 13`

### NOTE: Opset number

Some TensorFlow ops will fail to convert if the ONNX opset used is too low. **Use the largest opset compatible with your application.** For full conversion instructions, please refer to the [tf2onnx README](https://github.com/onnx/tensorflow-onnx#cli-reference).

## Verifying a Converted Model

Install onnxruntime with:

`pip install onnxruntime`

Test your model in python using the template below:

```python
import onnxruntime as ort
import numpy as np

# Change shapes and types to match model
input1 = np.zeros((1, 100, 100, 3), np.float32)

sess = ort.InferenceSession("dst/path/model.onnx")
# Set first argument of sess.run to None to use all model outputs in default order
# Input/output names are printed by the CLI and can be set with --rename-inputs and --rename-outputs
# If using the python API, names are determined from function arg names or TensorSpec names.
results_ort = sess.run(["output1", "output2"], {"input1": input1})

import tensorflow as tf
model = tf.saved_model.load("path/to/savedmodel")
results_tf = model(input1)

for ort_res, tf_res in zip(results_ort, results_tf):
    np.testing.assert_allclose(ort_res, tf_res, rtol=1e-5, atol=1e-5)

print("Results match")
```

## Viewing an ONNX Model

ONNX models can be viewed in the open-source [Netron](https://github.com/lutzroeder/Netron) tool. The tool can be used in-browser at [netron.app](https://netron.app/).

## Conversion Failures

If your model fails to convert please read the [README](https://github.com/onnx/tensorflow-onnx#readme) and [Troubleshooting guide](https://github.com/onnx/tensorflow-onnx/blob/master/Troubleshooting.md). If that fails feel free to [open an issue on GitHub](https://github.com/onnx/tensorflow-onnx/issues).  Contributions to tf2onnx are welcome!

## Next Steps

- [More tutorials: accelerate Tensorflow models](../inferencing/tensorflow.md#accelerate-tensorflow-model-inferencing-1)
