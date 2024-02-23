from enum import Enum

import onnx
from onnx import OperatorSetIdProto, TensorProto, helper


class Format(Enum):
    Format1 = (1,)
    Format2 = (2,)
    Format3 = 3


def generate_model(model_format, model_name, multi_output_add=False, add_output_in_graph_output=False, with_cast=False):
    nodes = []  # LayerNorm subgraph
    if with_cast:
        nodes.extend(
            [
                helper.make_node("Cast", ["ln_in"], ["c_out"], "cast", to=1),
                helper.make_node("ReduceMean", ["c_out"], ["rd1_out"], "reduce1", axes=[-1], keepdims=1),
                helper.make_node("Sub", ["c_out", "rd1_out"], ["sb1_out"], "sub1"),
                helper.make_node("Sub", ["c_out", "rd1_out"], ["sb2_out"], "sub2"),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("ReduceMean", ["ln_in"], ["rd1_out"], "reduce1", axes=[-1], keepdims=1),
                helper.make_node("Sub", ["ln_in", "rd1_out"], ["sb1_out"], "sub1"),
                helper.make_node("Sub", ["ln_in", "rd1_out"], ["sb2_out"], "sub2"),
            ]
        )
    nodes.extend(
        [  # LayerNorm subgraph
            helper.make_node("Pow", ["sb2_out", "pow_in_2"], ["pow_out"], "pow"),
            helper.make_node("ReduceMean", ["pow_out"], ["rd2_out"], "reduce2", axes=[-1], keepdims=1),
            helper.make_node("Add", ["rd2_out", "const_e12"], ["add1_out"], "add1"),
            helper.make_node("Sqrt", ["add1_out"], ["sqrt_out"], "sqrt"),
            helper.make_node("Div", ["sb1_out", "sqrt_out"], ["div_out"], "div1"),
            helper.make_node("Mul", ["gamma", "div_out"], ["mul_out"], "mul"),
            helper.make_node("Add", ["mul_out", "beta"], ["C"], "add0"),
        ]
    )

    initializers = [  # initializers
        helper.make_tensor("pow_in_2", TensorProto.FLOAT, [], [2]),
        helper.make_tensor("const_e12", TensorProto.FLOAT, [], [1e-12]),
        helper.make_tensor("gamma", TensorProto.FLOAT, [4], [1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor("beta", TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),
    ]

    if model_format is Format.Format1:
        nodes.extend(
            [
                helper.make_node("Add", ["A", "bias"], ["add3_out"], "add3"),
                helper.make_node("Add", ["add3_out", "B"], ["ln_in"], "add2"),
            ]
        )
        initializers.extend(
            [
                helper.make_tensor(
                    "bias", TensorProto.FLOAT16 if with_cast else TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]
                ),
            ]
        )
    elif model_format is Format.Format2:
        nodes.extend(
            [
                helper.make_node("Add", ["B", "bias"], ["add3_out"], "add3"),
                helper.make_node("Add", ["A", "add3_out"], ["ln_in"], "add2"),
            ]
        )
        initializers.extend(
            [
                helper.make_tensor(
                    "bias", TensorProto.FLOAT16 if with_cast else TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]
                ),
            ]
        )
    elif model_format is Format.Format3:
        nodes.extend(
            [
                helper.make_node("Add", ["A", "B"], ["ln_in"], "add2"),
            ]
        )

    if multi_output_add:
        neg_input = "ln_in" if model_format is Format.Format3 else "add3_out"
        nodes.extend([helper.make_node("Neg", [neg_input], ["neg_out"], "neg")])

    graph = helper.make_graph(
        nodes,
        "SkipLayerNorm_format3",  # name
        [  # inputs
            helper.make_tensor_value_info("A", TensorProto.FLOAT16 if with_cast else TensorProto.FLOAT, [16, 32, 4]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT16 if with_cast else TensorProto.FLOAT, [16, 32, 4]),
        ],
        [  # outputs
            helper.make_tensor_value_info("C", TensorProto.FLOAT, [16, 32, 4]),
        ],
        initializers,
    )

    if add_output_in_graph_output:
        extra_output = "ln_in" if model_format is Format.Format3 else "add3_out"
        graph.output.extend(
            [
                helper.make_tensor_value_info(
                    extra_output, TensorProto.FLOAT16 if with_cast else TensorProto.FLOAT, [16, 32, 4]
                )
            ]
        )

    onnxdomain = OperatorSetIdProto()
    onnxdomain.version = 12
    # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
    onnxdomain.domain = ""
    msdomain = OperatorSetIdProto()
    msdomain.version = 1
    msdomain.domain = "com.microsoft"
    opsets = [onnxdomain, msdomain]

    model = helper.make_model(graph, opset_imports=opsets)
    onnx.save(model, model_name)


def generate_skip_layer_norm(with_cast=False):
    suffix = "_with_cast" if with_cast else ""

    generate_model(Format.Format1, f"skip_layer_norm_format1{suffix}.onnx", with_cast=with_cast)
    generate_model(Format.Format2, f"skip_layer_norm_format2{suffix}.onnx", with_cast=with_cast)
    generate_model(Format.Format3, f"skip_layer_norm_format3{suffix}.onnx", with_cast=with_cast)
    generate_model(
        Format.Format1, f"skip_layer_norm_format1_partial{suffix}.onnx", multi_output_add=True, with_cast=with_cast
    )
    generate_model(
        Format.Format2, f"skip_layer_norm_format2_partial{suffix}.onnx", multi_output_add=True, with_cast=with_cast
    )
    generate_model(
        Format.Format3, f"skip_layer_norm_format3_no_fusion{suffix}.onnx", multi_output_add=True, with_cast=with_cast
    )
    generate_model(
        Format.Format1,
        f"skip_layer_norm_format1_graph_output{suffix}.onnx",
        add_output_in_graph_output=True,
        with_cast=with_cast,
    )
    generate_model(
        Format.Format2,
        f"skip_layer_norm_format2_graph_output{suffix}.onnx",
        add_output_in_graph_output=True,
        with_cast=with_cast,
    )
    generate_model(
        Format.Format3,
        f"skip_layer_norm_format3_graph_output{suffix}.onnx",
        add_output_in_graph_output=True,
        with_cast=with_cast,
    )


generate_skip_layer_norm(with_cast=False)
generate_skip_layer_norm(with_cast=True)
