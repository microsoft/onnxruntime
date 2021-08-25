# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sqlite3
import onnx
from onnx import numpy_helper

connection = sqlite3.connect('<path-to-sqldb-from-tracing>', detect_types=sqlite3.PARSE_DECLTYPES)

def convert_tensor_proto_to_numpy_array(blob):
    tensor_proto = onnx.TensorProto()
    tensor_proto.ParseFromString(blob)
    return numpy_helper.to_array(tensor_proto)

sqlite3.register_converter("TensorProto", convert_tensor_proto_to_numpy_array)

for step, name, value, device, producer, consumers in connection.execute(
        'Select Step, Name, Value, DeviceType, TracedProducer, TracedConsumers from Tensors'):
    print(step, name, value.shape, consumers)
