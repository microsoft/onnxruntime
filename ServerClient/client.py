import os
from onnx import *
from onnxruntime import *
import numpy as np

X_shape = [1,3,32,32]
Y_shape = [1,3,32,32]
Z_shape = [1,3,32,32,1,3,32,32]

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, X_shape)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, Y_shape)
Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, Y_shape)

remote = helper.make_node('RemoteCall',['X','Y'], ['Z'], domain='com.microsoft', uri='localhost:80/endpoint', key='')
graph = helper.make_graph([remote], 'graph', [X,Y], [Z])
model = onnx.helper.make_model(graph)
save(model, 'hybrid.onnx')

_=input(os.getpid())
sess = InferenceSession('hybrid.onnx', providers=['CPUExecutionProvider'])

x = np.random.random_sample(X_shape).astype(np.float32)
y = np.random.random_sample(Y_shape).astype(np.float32)

output = sess.run([], {'X':x,'Y':y})
print('done')


