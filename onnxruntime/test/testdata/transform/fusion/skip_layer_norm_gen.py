import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum


class Format(Enum):
    Format1 = 1,
    Format2 = 2,
    Format3 = 3


def GenerateModel(format, model_name, multi_output_add=False, add_output_in_graph_output=False):
    nodes = [  # LayerNorm subgraph
        helper.make_node("ReduceMean", ["ln_in"], ["rd1_out"], "reduce1", axes=[-1], keepdims=1),
        helper.make_node("Sub", ["ln_in", "rd1_out"], ["sb1_out"], "sub1"),
        helper.make_node("Sub", ["ln_in", "rd1_out"], ["sb2_out"], "sub2"),
        helper.make_node("Pow", ["sb2_out", "pow_in_2"], ["pow_out"], "pow"),
        helper.make_node("ReduceMean", ["pow_out"], ["rd2_out"], "reduce2", axes=[-1], keepdims=1),
        helper.make_node("Add", ["rd2_out", "const_e12"], ["add1_out"], "add1"),
        helper.make_node("Sqrt", ["add1_out"], ["sqrt_out"], "sqrt"),
        helper.make_node("Div", ["sb1_out", "sqrt_out"], ["div_out"], "div1"),
        helper.make_node("Mul", ["gamma", "div_out"], ["mul_out"], "mul"),
        helper.make_node("Add", ["mul_out", "beta"], ["C"], "add0"),
    ]

    initializers = [  # initializers
        helper.make_tensor('pow_in_2', TensorProto.FLOAT, [], [2]),
        helper.make_tensor('const_e12', TensorProto.FLOAT, [], [1e-12]),
        helper.make_tensor('gamma', TensorProto.FLOAT, [4], [1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('beta', TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),
    ]

    if format is Format.Format1:
        nodes.extend([
            helper.make_node("Add", ["A", "bias"], ["add3_out"], "add3"),
            helper.make_node("Add", ["add3_out", "B"], ["ln_in"], "add2"),
        ])
        initializers.extend([
            helper.make_tensor('bias', TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),
        ])
    elif format is Format.Format2:
        nodes.extend([
            helper.make_node("Add", ["B", "bias"], ["add3_out"], "add3"),
            helper.make_node("Add", ["A", "add3_out"], ["ln_in"], "add2"),
        ])
        initializers.extend([
            helper.make_tensor('bias', TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),
        ])
    elif format is Format.Format3:
        nodes.extend([
            helper.make_node("Add", ["A", "B"], ["ln_in"], "add2"),
        ])

    if multi_output_add:
        neg_input = "ln_in" if format is Format.Format3 else "add3_out"
        nodes.extend([helper.make_node("Neg", [neg_input], ["neg_out"], "neg")])

    graph = helper.make_graph(
        nodes,
        "SkipLayerNorm_format3",  #name
        [  # inputs
            helper.make_tensor_value_info('A', TensorProto.FLOAT, [16, 32, 4]),
            helper.make_tensor_value_info('B', TensorProto.FLOAT, [16, 32, 4]),
        ],
        [  # outputs
            helper.make_tensor_value_info('C', TensorProto.FLOAT, [16, 32, 4]),
        ],
        initializers)
    
    if add_output_in_graph_output:
        extra_output = "ln_in" if format is Format.Format3 else "add3_out"
        graph.output.extend([helper.make_tensor_value_info(extra_output, TensorProto.FLOAT, [16, 32, 4])])

    model = helper.make_model(graph)
    onnx.save(model, model_name)


GenerateModel(Format.Format1, 'skip_layer_norm_format1.onnx')
GenerateModel(Format.Format2, 'skip_layer_norm_format2.onnx')
GenerateModel(Format.Format3, 'skip_layer_norm_format3.onnx')
GenerateModel(Format.Format1, 'skip_layer_norm_format1_partial.onnx', multi_output_add = True)
GenerateModel(Format.Format2, 'skip_layer_norm_format2_partial.onnx', multi_output_add = True)
GenerateModel(Format.Format3, 'skip_layer_norm_format3_no_fusion.onnx', multi_output_add = True)

GenerateModel(Format.Format1, 'skip_layer_norm_format1_graph_output.onnx', add_output_in_graph_output = True)
GenerateModel(Format.Format2, 'skip_layer_norm_format2_graph_output.onnx', add_output_in_graph_output = True)
GenerateModel(Format.Format3, 'skip_layer_norm_format3_graph_output.onnx', add_output_in_graph_output = True)