import sys
import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum
import numpy as np

k = 1024
size = k
batch = 8
seqlen = 128

def GenerateModel(model_name):
    nodes = [
        # q nodes
        helper.make_node("MatMul", ["input", "weight"], ["output"], "MatMul000"),
    ]
    rng = np.random.default_rng()
    weight = rng.random((size * size), dtype=np.float32)
    initializers = [  # initializers
        helper.make_tensor('weight', TensorProto.FLOAT, [size, size], weight),
    ]

    graph = helper.make_graph(
        nodes,
        "MatMul000",  #name
        [  # inputs
            helper.make_tensor_value_info('input', TensorProto.FLOAT, [batch, seqlen, size])
        ],
        [  # outputs
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [batch, seqlen, size])
        ],
        initializers)

    model = helper.make_model(graph)
    onnx.save(model, model_name)

GenerateModel('matmul.onnx')

import onnxruntime as ort
import time

for round in range(20):
    sess_options = ort.SessionOptions()
    execution_providers = ['CUDAExecutionProvider']
    model = "matmul.onnx"
    ort_session = ort.InferenceSession(model, sess_options, providers=execution_providers)
    io_binding = ort_session.io_binding()
    io_binding.bind_cpu_input('input', np.ones((batch, seqlen, size), dtype=np.float32))
    io_binding.bind_output('output')
    time_onnx = 0
    # warm up
    ort_session.run_with_iobinding(io_binding)
    ort_session.run_with_iobinding(io_binding)
    ort_session.run_with_iobinding(io_binding)
    pretime = time.time()
    t = 30
    for i in range(30):
        ort_session.run_with_iobinding(io_binding)
    endtime = time.time()
    time_onnx += endtime - pretime
    print(str(time_onnx / t))
