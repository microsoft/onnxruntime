import math
import numpy
import os


def create_qordered_matmul_graph(m, n, k):
    from onnx import helper, numpy_helper, TensorProto

    nodes = [
        helper.make_node(
            'QOrderedMatMul',
            inputs=['A', 'scale_A', 'B', 'scale_B', 'scale_Y'],
            outputs=['Y'],
            name='qordered_matmul_0',
            domain='com.microsoft',
            order_A=2,
            order_B=3,
            order_Y=2,
        ),
    ]

    initializers = [
        numpy_helper.from_array(numpy.array(0.007874015718698502, dtype='float32'), name='scale_A'),
        numpy_helper.from_array(numpy.array(0.007874015718698502, dtype='float32'), name='scale_B'),
        numpy_helper.from_array(numpy.array(0.007874015718698502, dtype='float32'), name='scale_Y'),
    ]

    graph = helper.make_graph(nodes, "QOrderedMatMulGraph", [
        helper.make_tensor_value_info('A', TensorProto.INT8, [m, k]),
        helper.make_tensor_value_info('B', TensorProto.INT8, [k, n]),
    ], [
        helper.make_tensor_value_info('Y', TensorProto.INT8, [m, n]),
    ], initializers)

    model = helper.make_model(graph=graph)
    return model.SerializeToString()


m = 768
n = 768
k = 32

onnx_model_str = create_qordered_matmul_graph(m, n, k)

from onnxruntime import SessionOptions, InferenceSession
sess_options = SessionOptions()
ort_session = InferenceSession(onnx_model_str, sess_options, providers=['CUDAExecutionProvider'])

ort_inputs = {
    'A' : numpy.random.randint(-127, 128, [m, k], dtype=numpy.int8),
    'B' : numpy.random.randint(-127, 128, [k, n], dtype=numpy.int8)
}

ort_output = ort_session.run(None, ort_inputs)
print(ort_output)