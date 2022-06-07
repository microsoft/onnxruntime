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
<<<<<<<< HEAD:docs/api/python/downloads/3e23fa9ebb26f4728ee8426ed7da0f63/plot_backend.py
from onnxruntime import datasets
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
import onnxruntime.backend as backend
from onnx import load

########################################
# The device depends on how the package was compiled,
# GPU or CPU.
from onnxruntime import get_device
========
from onnx import load

import onnxruntime.backend as backend

########################################
# The device depends on how the package was compiled,
# GPU or CPU.
from onnxruntime import datasets, get_device
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

>>>>>>>> 4296968f2057743d43ebf6f3aebd48571fe3f9d7:docs/api/python/sources/inference/examples/plot_backend.py
device = get_device()

name = datasets.get_example("logreg_iris.onnx")
model = load(name)

rep = backend.prepare(model, device)
x = np.array([[-1.0, -2.0]], dtype=np.float32)
try:
    label, proba = rep.run(x)
    print("label={}".format(label))
    print("probabilities={}".format(proba))
except (RuntimeError, InvalidArgument) as e:
    print(e)

########################################
# The backend can also directly load the model
# without using *onnx*.

rep = backend.prepare(name, device)
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
