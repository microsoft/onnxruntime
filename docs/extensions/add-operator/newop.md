---
title: Add a New C++ Operator
description: Instructions to add a new custom operator using C++
parent: Add Operators
grand_parent: Extensions
nav_order: 2
---
# Create a Custom Operator from Scratch

Before implementing a custom operator, you need an ONNX model with one or more ORT custom operators, created by ONNX converters, [ONNX-Script](https://github.com/microsoft/onnx-script), or [ONNX model API](https://onnx.ai/onnx/api/helper.html) and etc..


## 1. Quick verification with PythonOp (optional)

Before you actually develop a custom operator for your use case, if you want to quickly verify the ONNX model with Python, you can wrap the custom operator with **[PyOp](./pyop.md)**.

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
- Check [build-package](../build.md) on how to build the different language package to be used for production.

Please check the [contribution](../index.md#contributing) to see if it is possible to contribute the custom operator to onnxruntime-extensions.