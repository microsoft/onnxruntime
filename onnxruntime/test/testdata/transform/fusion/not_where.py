import onnx
from onnx import helper
from onnx import TensorProto, OperatorSetIdProto
from enum import Enum

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = 'com.microsoft'

opsets.append(msdomain)
kwargs={}
kwargs['opset_imports'] = opsets

def GenerateModel(model_name):
    nodes = [  # subgraph
        # float
        helper.make_node("Not", ["X"], ["not_X"], "not_1"),
        helper.make_node("Where", ["not_X", "v0", "v1"], ["Y"], "where_1")
    ]

    inputs = [  # inputs
            helper.make_tensor_value_info('X', TensorProto.BOOL, ['M', 'K']),
        ]

    initializers = [
            helper.make_tensor('v0', TensorProto.FLOAT, [1], [1.0]),
            helper.make_tensor('v1', TensorProto.FLOAT, [1], [-1.0]),
        ]

    graph = helper.make_graph(
        nodes,
        "NotWhere",  #name
        inputs,
        [  # outputs
            helper.make_tensor_value_info('Y', TensorProto.FLOAT, ['M', 'K']),
        ],
        initializers)

    model = helper.make_model(graph, **kwargs)
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModel('not_where.onnx')