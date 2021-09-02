# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""

.. _l-example-profiling:

Profile the execution of a simple model
=======================================

*ONNX Runtime* can profile the execution of the model.
This example shows how to interpret the results.
"""
import onnx
import onnxruntime as rt
import numpy
from onnxruntime.datasets import get_example


def change_ir_version(filename, ir_version=6):
    "onnxruntime==1.2.0 does not support opset <= 7 and ir_version > 6"
    with open(filename, "rb") as f:
        model = onnx.load(f)
    model.ir_version = 6
    if model.opset_import[0].version <= 7:
        model.opset_import[0].version = 11
    return model




#########################
# Let's load a very simple model and compute some prediction.

example1 = get_example("mul_1.onnx")
onnx_model = change_ir_version(example1)
onnx_model_str = onnx_model.SerializeToString()
sess = rt.InferenceSession(onnx_model_str)
input_name = sess.get_inputs()[0].name

x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
res = sess.run(None, {input_name: x})
print(res)

#########################
# We need to enable to profiling
# before running the predictions.

options = rt.SessionOptions()
options.enable_profiling = True
sess_profile = rt.InferenceSession(onnx_model_str, options)
input_name = sess.get_inputs()[0].name

x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)

sess.run(None, {input_name: x})
prof_file = sess_profile.end_profiling()
print(prof_file)

###########################
# The results are stored un a file in JSON format.
# Let's see what it contains.
import json
with open(prof_file, "r") as f:
    sess_time = json.load(f)
import pprint
pprint.pprint(sess_time)


    
