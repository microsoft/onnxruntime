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
        helper.make_node("Cast", ["A"], ["cast1"], "cast_1", to=11),

        helper.make_node("IsInf", ["cast1"], ["IsInf_out"], "is_inf"),

        helper.make_node("Cast", ["IsInf_out"], ["cast2"], "cast_2", to=7),

        helper.make_node("ReduceSum", ["cast2"], ["reduced"], "reduction", keepdims=0),

        helper.make_node("Greater", ["reduced", "one"], ["Y"], "output")
    ]

    inputs = [  # inputs
            helper.make_tensor_value_info('A', TensorProto.FLOAT16, ['M', 'K']),
        ]

    initializers = [
        helper.make_tensor('one', TensorProto.INT64, [1], [1])]

    graph = helper.make_graph(
        nodes,
        "IsInfReduceSum",  #name
        inputs,
        [  # outputs
            helper.make_tensor_value_info('Y', TensorProto.BOOL, [1]),
        ],
        initializers)

    model = helper.make_model(graph, **kwargs)
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModel('isinf_reducesum.onnx')