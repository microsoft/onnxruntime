import numpy

import onnxruntime
from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_pybind11_state import RunOptions

# usage:
# 1. build onnxruntime: ./build.sh --parallel --skip_tests --build_shared_lib --build_wheel
# 2. build external EP:
# cd samples/customEP2/build
# cmake --build .
# 3. run this script: python3 test_customEp2.py

model_path = '/bert_ort/leca/models/Max.onnx'
shared_lib_path = '/bert_ort/leca/code/onnxruntime2/samples/customEP2/build/libcustomep2.so'

onnxruntime.load_execution_provider_info('customEp2', shared_lib_path)
print('Load External EP success')
session = onnxruntime.InferenceSession(model_path,
    providers=['customEp2'], provider_options=[{'int_property':'3', 'str_property':'strval'}])
input0 = numpy.zeros(3, dtype=numpy.float32)
input0[0], input0[1], input0[2] = 7, 0, 1
input1 = numpy.zeros(3, dtype=numpy.float32)
input1[0], input1[1], input1[2] = 4, 9, 2
input2 = numpy.zeros(3, dtype=numpy.float32)
input2[0], input2[1], input2[2] = 5, 3, 8
y = session.run(['output'], {'input0': input0, 'input1': input1, 'input2': input2})
print('y:')
print(y)

