# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""

.. _l-example-backend-api:

ONNX Runtime Backend for ONNX
=============================

*ONNX Runtime* extends the 
`onnx backend API <https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md>`_
to run predictions using this runtime.
Let's use the API to compute the prediction
of a simple logistic regression model.
"""
import numpy as np
from onnxruntime import datasets
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
import onnxruntime.backend as backend
from onnx import load

name = datasets.get_example("logreg_iris.onnx")
model = load(name)

rep = backend.prepare(model, 'CPU')
x = np.array([[-1.0, -2.0]], dtype=np.float32)
try:
    label, proba = rep.run(x)
    print("label={}".format(label))
    print("probabilities={}".format(proba))
except (RuntimeError, InvalidArgument) as e:
    print(e)

########################################
# The device depends on how the package was compiled,
# GPU or CPU.
from onnxruntime import get_device
print(get_device())

########################################
# The backend can also directly load the model
# without using *onnx*.

rep = backend.prepare(name, 'CPU')
x = np.array([[-1.0, -2.0]], dtype=np.float32)
try:
    label, proba = rep.run(x)
    print("label={}".format(label))
    print("probabilities={}".format(proba))
except (RuntimeError, InvalidArgument) as e:
    print(e)

#######################################
# The backend API is implemented by other frameworks
# and makes it easier to switch between multiple runtimes
# with the same API.
