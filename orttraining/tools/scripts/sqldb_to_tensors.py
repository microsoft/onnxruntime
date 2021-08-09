import sqlite3
import onnx
from onnx import numpy_helper

connection = sqlite3.connect('ort-trace.db', 
    detect_types=sqlite3.PARSE_DECLTYPES)

def convert_tensor_proto_to_numpy_array(blob):
    tensor_proto = onnx.TensorProto()
    tensor_proto.ParseFromString(blob)
    return numpy_helper.to_array(tensor_proto)

sqlite3.register_converter("TensorProto", convert_tensor_proto_to_numpy_array)

for step, name, value, device in connection.execute('select step, name, value, device from tensors'):
    print(step, name, value.shape, device)
