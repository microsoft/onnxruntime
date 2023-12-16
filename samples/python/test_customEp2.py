import os
from sys import platform

import numpy as np

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

#model_path = './identity.onnx'
model_path = '/bert_ort/leca/models/Detection/model.onnx'
shared_lib_path = '/onnxruntime/samples/customEP2/build/Debug/customep2.dll'
if platform == 'linux' or platform == 'linux2':
    #shared_lib_path = '../customEP2/build/libcustomep2.so'
    shared_lib_path = '../openvino/build/libcustom_openvino.so'

#onnxruntime.load_execution_provider_info('customEp2', shared_lib_path)
onnxruntime.load_execution_provider_info('openvino', shared_lib_path)
#_ = input(os.getpid())

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

session = onnxruntime.InferenceSession(model_path, sess_options,
#    providers=['customEp2'], provider_options=[{'int_property':'3', 'str_property':'strval'}])
    providers=['openvino'], provider_options=[{'number_of_threads':'8'}])
#x = np.random.rand(6).astype(np.float32)
#y = session.run(None, {'X': x})
y = session.run(None, {'input': np.random.rand(1,3,5,5).astype(np.float32)}) # Detection model
print(y)
#_ = input('done')
