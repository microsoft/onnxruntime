# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _l-example-common-error:

Common errors with onnxruntime
==============================

This example looks into several common situations
in which *onnxruntime* does not return the model
prediction but raises an exception instead.
It starts by loading the model trained in example
:ref:`l-logreg-example` which produced a logistic regression
trained on *Iris* datasets. The model takes
a vector of dimension 2 and returns a class among three.
"""
import numpy

import onnxruntime as rt
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
from onnxruntime.datasets import get_example

example2 = get_example("logreg_iris.onnx")
sess = rt.InferenceSession(example2, providers=rt.get_available_providers())

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

#############################
# The first example fails due to *bad types*.
# *onnxruntime* only expects single floats (4 bytes)
# and cannot handle any other kind of floats.

try:
    x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float64)
    sess.run([output_name], {input_name: x})
except Exception as e:
    print("Unexpected type")
    print(f"{type(e)}: {e}")

#########################
# The model fails to return an output if the name
# is misspelled.

try:
    x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
    sess.run(["misspelled"], {input_name: x})
except Exception as e:
    print("Misspelled output name")
    print(f"{type(e)}: {e}")

###########################
# The output name is optional, it can be replaced by *None*
# and *onnxruntime* will then return all the outputs.

x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
try:
    res = sess.run(None, {input_name: x})
    print("All outputs")
    print(res)
except (RuntimeError, InvalidArgument) as e:
    print(e)

#########################
# The same goes if the input name is misspelled.

try:
    x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
    sess.run([output_name], {"misspelled": x})
except Exception as e:
    print("Misspelled input name")
    print(f"{type(e)}: {e}")

#########################
# *onnxruntime* does not necessarily fail if the input
# dimension is a multiple of the expected input dimension.

for x in [
    numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32),
    numpy.array([[1.0, 2.0, 3.0, 4.0]], dtype=numpy.float32),
    numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32),
    numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32),
    numpy.array([[1.0, 2.0, 3.0]], dtype=numpy.float32),
]:
    try:
        r = sess.run([output_name], {input_name: x})
        print(f"Shape={x.shape} and predicted labels={r}")
    except (RuntimeError, InvalidArgument) as e:
        print(f"ERROR with Shape={x.shape} - {e}")

for x in [
    numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32),
    numpy.array([[1.0, 2.0, 3.0, 4.0]], dtype=numpy.float32),
    numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32),
    numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32),
    numpy.array([[1.0, 2.0, 3.0]], dtype=numpy.float32),
]:
    try:
        r = sess.run(None, {input_name: x})
        print(f"Shape={x.shape} and predicted probabilities={r[1]}")
    except (RuntimeError, InvalidArgument) as e:
        print(f"ERROR with Shape={x.shape} - {e}")

#########################
# It does not fail either if the number of dimension
# is higher than expects but produces a warning.

for x in [
    numpy.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=numpy.float32),
    numpy.array([[[1.0, 2.0, 3.0]]], dtype=numpy.float32),
    numpy.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=numpy.float32),
]:
    try:
        r = sess.run([output_name], {input_name: x})
        print(f"Shape={x.shape} and predicted labels={r}")
    except (RuntimeError, InvalidArgument) as e:
        print(f"ERROR with Shape={x.shape} - {e}")
