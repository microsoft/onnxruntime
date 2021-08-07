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
        helper.make_node("Div", ["float_1", "A"], ["div_1"], "div_1"),
        helper.make_node("Mul", ["div_1", "B"], ["mul_1"], "mul_1"),
        helper.make_node("Cast", ["mul_1"], ["cast_1"], "cast_1", to=10),
        # float_16
        helper.make_node("Div", ["float16_1", "cast_1"], ["div_2"], "div_2"),
        helper.make_node("Mul", ["C", "div_2"], ["mul_2"], "mul_2"),
        helper.make_node("Cast", ["mul_2"], ["cast_2"], "cast_2", to=7),
        # int64
        helper.make_node("Div", ["int64_1", "cast_2"], ["div_3"], "div_3"),
        helper.make_node("Mul", ["D", "div_3"], ["mul_3"], "mul_3"),
        helper.make_node("Identity", ["mul_3"], ["Y"], "output"),
        # div has >1 consumers
        helper.make_node("Div", ["float_1", "A"], ["div_4"], "div_4"),
        helper.make_node("Mul", ["div_4", "B"], ["mul_4"], "mul_4"),
        # div is graph output
        helper.make_node("Div", ["float_1", "div_4"], ["div_5"], "div_5"),
        helper.make_node("Mul", ["div_5", "B"], ["mul_5"], "mul_5"),
    ]

    inputs = [  # inputs
            helper.make_tensor_value_info('A', TensorProto.FLOAT, ['M', 'K']),
            helper.make_tensor_value_info('B', TensorProto.FLOAT, ['M', 'K']),
            helper.make_tensor_value_info('C', TensorProto.FLOAT16, ['M', 'K']),
            helper.make_tensor_value_info('D', TensorProto.INT64, ['M', 'K']),
        ]

    initializers = [
            helper.make_tensor('float_1', TensorProto.FLOAT, [1], [1.0]),
            helper.make_tensor('float16_1', TensorProto.FLOAT16, [1], [15360]), # 15360 is the fp16 representation of 1.f
            helper.make_tensor('int64_1', TensorProto.INT64, [1], [1]),
        ]

    graph = helper.make_graph(
        nodes,
        "DivMul",  #name
        inputs,
        [  # outputs
            helper.make_tensor_value_info('Y', TensorProto.INT64, ['M', 'K']),
            helper.make_tensor_value_info('div_5', TensorProto.FLOAT, ['M', 'K']),
        ],
        initializers)

    model = helper.make_model(graph, **kwargs)
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModel('div_mul.onnx')