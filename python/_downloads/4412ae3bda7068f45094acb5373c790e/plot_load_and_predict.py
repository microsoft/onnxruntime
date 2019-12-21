# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _l-example-simple-usage:

Load and predict with ONNX Runtime and a very simple model
==========================================================

This example demonstrates how to load a model and compute
the output for an input vector. It also shows how to
retrieve the definition of its inputs and outputs.
"""

import onnxruntime as rt
import numpy
from onnxruntime.datasets import get_example

#########################
# Let's load a very simple model.
# The model is available on github `onnx...test_sigmoid <https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node/test_sigmoid>`_.

example1 = get_example("sigmoid.onnx")
sess = rt.InferenceSession(example1)

#########################
# Let's see the input name and shape.

input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

#########################
# Let's see the output name and shape.

output_name = sess.get_outputs()[0].name
print("output name", output_name)  
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)

#########################
# Let's compute its outputs (or predictions if it is a machine learned model).

import numpy.random
x = numpy.random.random((3,4,5))
x = x.astype(numpy.float32)
res = sess.run([output_name], {input_name: x})
print(res)
