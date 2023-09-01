---
title: Add Operators
description: Instructions to add a new custom operator
parent: Extensions
nav_order: 2
---

# Creating custom operators using Python functions

Custom operators are a powerful feature in ONNX Runtime that allows users to extend the functionality of the runtime by implementing their own operators to perform specific operations not available in the standard ONNX operator set.

In this document, we will introduce how to create a custom operator using Python functions and integrate it into ONNX Runtime for inference.


## Step 1: Define the Python function for the custom operator
Start by defining the Python function that will serve as the implementation for your custom operator. Ensure that the function is compatible with the input and output tensor shapes you expect for your custom operator.
the Python decorator @onnx_op will convert the function to be a custom operator implementation. The following is example we create a function for a tokenizer 

```Python
@onnx_op(op_type="GPT2Tokenizer",
            inputs=[PyCustomOpDef.dt_string],
            outputs=[PyCustomOpDef.dt_int64, PyCustomOpDef.dt_int64],
            attrs={"padding_length": PyCustomOpDef.dt_int64})
def bpe_tokenizer(s, **kwargs):
    padding_length = kwargs["padding_length"]
    input_ids, attention_mask = cls.tokenizer.tokenizer_sentence([s[0]], padding_length)
    return input_ids, attention_mask
```
Because ONNXRuntimme needs the custom operator schema on loading a model, please specify them by onnx_op arguments. Also 'attrs' is needed if there are attributes for the ONNX node, which can be dict that mapping from its name to its type, or be a list if all types are string only.

## Step 2: Create an ONNX model with the custom operator
Now that the custom operator is registered with ONNX Runtime, you can create an ONNX model that utilizes it. You can either modify an existing ONNX model to include the custom operator or create a new one from scratch.

To create a new ONNX model with the custom operator, you can use the ONNX Python API. Here is an example: [test_pyops.py](https://github.com/microsoft/onnxruntime-extensions/blob/main/test/test_pyops.py)

# Create a Custom Operator from Scratch in C++

Before implementing a custom operator, you need an ONNX model with one or more ORT custom operators, created by ONNX converters, such as [ONNX-Script](https://github.com/microsoft/onnx-script), [ONNX model API](https://onnx.ai/onnx/api/helper.html), etc.


## 1. Quick verification with PythonOp (optional)

Before you actually develop a custom operator for your use case, if you want to quickly verify the ONNX model with Python, you can wrap the custom operator with Python functions as described above.

```python
import numpy
from onnxruntime_extensions import PyOp, onnx_op

# Implement the CustomOp by decorating a function with onnx_op
@onnx_op(op_type="Inverse", inputs=[PyOp.dt_float])
def inverse(x):
    # the user custom op implementation here:
    return numpy.linalg.inv(x)

# Run the model with this custom op
# model_func = PyOrtFunction(model_path)
# outputs = model_func(inputs)
# ...
```

## 2. Generate the C++ template code of the Custom operator from the ONNX Model (optional)
    python -m onnxruntime-extensions.cmd --cpp-gen <model_path> <repository_dir>`
If you are familiar with the ONNX model detail, you create the custom operator C++ classes directly.


## 3. Implement the CustomOp Kernel Compute method in the generated C++ files.
the custom operator kernel C++ code example can be found [operators](https://github.com/microsoft/onnxruntime-extensions/tree/main/operators) folder, like [gaussian_blur](https://github.com/microsoft/onnxruntime-extensions/blob/main/operators/cv2/imgproc/gaussian_blur.hpp). All C++ APIs that can be used in the kernel implementation are listed below

* [ONNXRuntime Custom API docs](https://onnxruntime.ai/docs/api/c/struct_ort_custom_op.html)
* the third libraries API docs integrated in ONNXRuntime Extensions the can be used in C++ code
    - OpenCV API docs https://docs.opencv.org/4.x/
    - Google SentencePiece Library docs https://github.com/google/sentencepiece/blob/master/doc/api.md
    - dlib(matrix and ML library) C++ API docs http://dlib.net/algorithms.html
    - BlingFire Library https://github.com/microsoft/BlingFire
    - Google RE2 Library https://github.com/google/re2/wiki/CplusplusAPI
    - JSON library https://json.nlohmann.me/api/basic_json/

## 3. Build and Test
- The unit tests can be implemented as Python or C++, check [test](https://github.com/microsoft/onnxruntime-extensions/tree/main/test) folder for more examples
- Check [build-package](./build.md) on how to build the different language package to be used for production.

Please check the [contribution](./index.md#contributing) to see if it is possible to contribute the custom operator to onnxruntime-extensions.