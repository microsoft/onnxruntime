import numpy

import onnxruntime
from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_pybind11_state import RunOptions

# usage:
# 1. build onnxruntime: ./build.sh --parallel --skip_tests --build_shared_lib --build_wheel --use_intree
# 2. run this script: python3 test_intreeEp.py

#model_path = '/bert_ort/leca/models/Relu.onnx'
model_path = '/home/leca/models/Relu.onnx'

session = onnxruntime.InferenceSession(model_path,
    providers=['InTreeExecutionProvider'], provider_options=[{'int_property':'3', 'str_property':'strval'}])
x = numpy.zeros(4, dtype=numpy.float32)
x[0], x[1], x[2], x[3] = -3, 5, -2, 4
y = session.run(['graphOut'], {'x': x})
print('y:')
print(y)

