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
        helper.make_node("Not", ["X"], ["not_X_1"], "not_1"),
        helper.make_node("Where", ["not_X_1", "v0", "v1"], ["Y1"], "where_1"),
        helper.make_node("Not", ["not_X_1"], ["x"], "not_2"),
        helper.make_node("Identity", ["v0"], ["v0_edge"], "identity_v0"),
        helper.make_node("Identity", ["v1"], ["v1_edge"], "identity_v1"),
        helper.make_node("Where", ["x", "v0_edge", "v1_edge"], ["Y2"], "where_2"),
        helper.make_node("Not", ["X"], ["not_X_2"], "not_3"),
        helper.make_node("Where", ["not_X_2", "v0", "v1"], ["Y3"], "where_3"),
        helper.make_node("Not", ["X"], ["not_X_3"], "not_4"),
        helper.make_node("Where", ["not_X_3", "v0", "v1"], ["Y4"], "where_4"),
        helper.make_node("Where", ["not_X_3", "v0", "v1"], ["Y5"], "where_5"),
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
            helper.make_tensor_value_info('not_X_2', TensorProto.BOOL, ['M', 'K']),
            helper.make_tensor_value_info('Y1', TensorProto.FLOAT, ['M', 'K']),
            helper.make_tensor_value_info('Y2', TensorProto.FLOAT, ['M', 'K']),
            helper.make_tensor_value_info('Y3', TensorProto.FLOAT, ['M', 'K']),
            helper.make_tensor_value_info('Y4', TensorProto.FLOAT, ['M', 'K']),
            helper.make_tensor_value_info('Y5', TensorProto.FLOAT, ['M', 'K']),
        ],
        initializers)

    model = helper.make_model(graph, **kwargs)
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModel('not_where.onnx')