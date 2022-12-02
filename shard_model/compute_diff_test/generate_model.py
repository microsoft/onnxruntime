import onnx
from onnx import helper, shape_inference
from onnx import TensorProto, OperatorSetIdProto
import onnxruntime
import numpy as np

def generate_model(nodes, input_tensors, output_tensors, init_tensors = None, name=None, value_info=None):
    if name is None:
        name = 'onnx-model'
    graph_def = helper.make_graph(nodes, name, input_tensors, output_tensors, init_tensors, value_info=value_info)

    opsets = []
    onnxdomain = OperatorSetIdProto()
    onnxdomain.version = 17
    onnxdomain.domain = ""  # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
    opsets.append(onnxdomain)

    msdomain = OperatorSetIdProto()
    msdomain.version = 1
    msdomain.domain = "com.microsoft"
    opsets.append(msdomain)

    kwargs = {}
    kwargs["opset_imports"] = opsets

    model_def = helper.make_model(graph_def, producer_name='onnxruntime-test', **kwargs)
    return model_def


def coll(op_name, s_input, s_output, **kwargs):
    node = helper.make_node(op_name,
                              inputs=[s_input],
                              outputs=[s_output],
                              domain='com.microsoft',
                              **kwargs)

    return node

def matmul(s_in_a, s_in_b, s_output):
    return helper.make_node('MatMul', [s_in_a, s_in_b], [s_output])

def reshape(in_a, shape, shape_name, out):
    shape = np.array(shape, dtype=np.int64)
    shape = onnx.numpy_helper.from_array(shape, shape_name) 
    node = helper.make_node('Reshape', [in_a, shape_name], [out])
    return node, shape

def transpose(in_a, perm, out):
    node = helper.make_node('Transpose', [in_a], [out], perm=perm)
    return node

