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

#model_path = '/bert_ort/leca/models/CustomOpTwo.onnx'
#model_path = '/bert_ort/leca/models/Detection/model.onnx'
#model_path = '/bert_ort/leca/models/Relu.onnx'
# model_path = '/home/leca/models/Relu.onnx'
#shared_lib_path = '/bert_ort/leca/code/onnxruntime2/samples/customEP2/build/libcustomep2.so'

'''
shared_lib_path = '/home/leca/code/onnxruntime/samples/customEP2/build/libcustomep2.so'
onnxruntime.load_execution_provider_info('customEp2', shared_lib_path)
print('Load External EP success')
session = onnxruntime.InferenceSession(model_path,
    providers=['customEp2'], provider_options=[{'int_property':'3', 'str_property':'strval'}])
x = numpy.zeros(4, dtype=numpy.float32)
x[0], x[1], x[2], x[3] = -3, 5, -2, 4
y = session.run(['graphOut'], {'x': x})
print('y:')
print(y)
'''

model_path = '/onnxruntime/samples/python/identity.onnx'
shared_lib_path = '/onnxruntime/samples/customEP2/build/Debug/customep2.dll'

onnxruntime.load_execution_provider_info('customEp2', shared_lib_path)
_ = input(os.getpid())

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

session = onnxruntime.InferenceSession(model_path, sess_options,
    providers=['customEp2'], provider_options=[{'int_property':'3', 'str_property':'strval'}])
x = np.random.rand(6).astype(np.float32)
y = session.run(None, {'X': x})
print(y)
_ = input('done')