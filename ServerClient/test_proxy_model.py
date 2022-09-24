import os
from onnx import *
from onnxruntime import *
import numpy as np
from create_proxy_model import create_proxy_model

X_shape = [1,3,32,32]
Y_shape = [1,3,32,32]
Z_shape = [1,3,32,32,1,3,32,32]

'''
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, X_shape)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, Y_shape)
Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, Z_shape)

#remote = helper.make_node('RemoteCall',['X','Y'], ['Z'], domain='com.microsoft', uri='localhost:80/endpoint', key='')
remote = helper.make_node('RemoteCall',['X','Y'], ['Z'], domain='com.microsoft')
graph = helper.make_graph([remote], 'graph', [X,Y], [Z])
model = onnx.helper.make_model(graph)
save(model, 'hybrid.onnx')

'''

model_name = 'hybrid.onnx'
input_names = ['X','Y']
input_types = [TensorProto.FLOAT, TensorProto.FLOAT]
input_shapes = [X_shape,Y_shape]
output_names = ['Z']
output_types = [TensorProto.FLOAT]
output_shapes = [Z_shape]
    
create_proxy_model(model_name, input_names, input_types, input_shapes, output_names, output_types, output_shapes)
_ = input(os.getpid())

providers_options = [
    ('AzureExecutionProvider', {
        'end_point': 'localhost:80/endpoint',
        'access_token': ''
    }),
    'CPUExecutionProvider',
]
sess = InferenceSession('hybrid.onnx', providers=providers_options)

x = np.random.random_sample(X_shape).astype(np.float32)
y = np.random.random_sample(Y_shape).astype(np.float32)

output = sess.run(None, {'X':x,'Y':y})
print (output)
print('done')


