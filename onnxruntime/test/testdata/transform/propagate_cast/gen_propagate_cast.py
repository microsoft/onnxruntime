import onnx
from onnx import helper
from onnx import TensorProto
from onnx import OperatorSetIdProto
import itertools

onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
# The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
onnxdomain.domain = ""
msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"
opsets = [onnxdomain, msdomain]

# expect type to be either TensorProto.FLOAT or TensorProto.FLOAT16
def type_to_string(type):
    return "float" if type ==TensorProto.FLOAT else "float16"

def save(model_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(
        nodes,
        "CastPropagateTest",
        inputs, outputs, initializers)

    model = helper.make_model(
        graph, opset_imports=opsets, producer_name="onnxruntime-test")

    onnx.save(model, model_path + ".onnx")

def gen_fuse_back2back_casts(model_path):

    for (type1, type2) in list(itertools.product([TensorProto.FLOAT, TensorProto.FLOAT16], repeat=2)):

        nodes = [
            helper.make_node(
                "MatMul",
                ["input_0", "input_1"],
                ["product"],
                "MatMul_0"),
            helper.make_node(
                "Cast",
                ["product"],
                ["product_cast"],
                "Cast_0",
                to = type1),
            helper.make_node(
                "Cast",
                ["product_cast"],
                ["output"],
                "Cast_1",
                to = type2)
        ]
        input_type = type2 if type1 != type2 else (TensorProto.FLOAT16 if type1 == TensorProto.FLOAT else  TensorProto.FLOAT)
        output_type = input_type if type1 != type2 else (TensorProto.FLOAT16 if input_type == TensorProto.FLOAT else  TensorProto.FLOAT)
        inputs = [
            helper.make_tensor_value_info(
                "input_0", input_type, ['M', 'K']),
            helper.make_tensor_value_info(
                "input_1", input_type, ['K', 'N'])
        ]

        outputs = [
            helper.make_tensor_value_info(
                "output",  output_type, ['M', 'N']),
        ]

        save(model_path + "_" + type_to_string(type1) + "_" + type_to_string(type2), nodes, inputs, outputs, [])

def gen_fuse_sibling_casts(model_path):

    for (type1, type2) in list(itertools.product([TensorProto.FLOAT, TensorProto.FLOAT16], repeat=2)):
        input_type = type2 if type1 != type2 else (TensorProto.FLOAT16 if type1 == TensorProto.FLOAT else  TensorProto.FLOAT)
        nodes = [
            helper.make_node(
                "MatMul",
                ["input_0", "input_1"],
                ["product"],
                "MatMul_0"),
            helper.make_node(
                "Cast",
                ["product"],
                ["cast_0_output"],
                "Cast_0",
                to = type1),
            helper.make_node(
                "Identity",
                ["cast_0_output"],
                ["output_0"],
                "Identity_0"),
            helper.make_node(
                "Cast",
                ["product"],
                ["cast_1_output"],
                "Cast_1",
                to = type2),
            helper.make_node(
                "Identity",
                ["cast_1_output"],
                ["output_1"],
                "Identity_1")
        ]

        inputs = [
            helper.make_tensor_value_info(
                "input_0", input_type, ['M', 'K']),
            helper.make_tensor_value_info(
                "input_1", input_type, ['K', 'N'])
        ]

        outputs = [
            helper.make_tensor_value_info(
                "output_0", type1, ['M', 'N']),
            helper.make_tensor_value_info(
                "output_1", type2, ['M', 'N'])
        ]

        save(model_path + "_" + type_to_string(type1) + "_" + type_to_string(type2), nodes, inputs, outputs, [])

def gen_propagate_cast_test_model(model_path, transpose_inputs, transpose_output, insert_input_casts, insert_output_cast, is_float16):
    nodes = [
        helper.make_node(
            "MatMul",
            ["transpose_output_0" if transpose_inputs else ("cast_output_0" if insert_input_casts else "input_0"),
             "transpose_output_1" if transpose_inputs else ("cast_output_1" if insert_input_casts else "input_1")],
            ["product"],
            "MatMul_0")
    ]

    if insert_output_cast:
        output_cast_type = TensorProto.FLOAT if is_float16 else TensorProto.FLOAT16
        nodes.append(helper.make_node(
            "Cast",
             ["transpose_output_2" if transpose_output else "product"],
             ["cast_output_2"],
             "Cast_2",
             to = output_cast_type))

    if insert_input_casts:
        input_cast_type = TensorProto.FLOAT16 if is_float16 else TensorProto.FLOAT
        nodes.extend([helper.make_node(
            "Cast",
            ["input_0"],
            ["cast_output_0"],
            "Cast_0",
            to = input_cast_type),
        helper.make_node(
            "Cast",
            ["input_1"],
            ["cast_output_1"],
            "Cast_1",
            to = input_cast_type)])

    if transpose_inputs:
        nodes.extend([helper.make_node("Transpose", ["cast_output_0" if insert_input_casts else "input_0"], ["transpose_output_0"], "Transpose_0"),
                      helper.make_node("Transpose", ["cast_output_1" if insert_input_casts else "input_1"], ["transpose_output_1"], "Transpose_1")])

    if transpose_output:
        nodes.extend([helper.make_node("Transpose", ["product"], ["transpose_output_2"], "Transpose_2")])

    input_type = (TensorProto.FLOAT if insert_input_casts else TensorProto.FLOAT16) if is_float16 else (TensorProto.FLOAT16 if insert_input_casts else TensorProto.FLOAT)
    output_type = (TensorProto.FLOAT if insert_output_cast else TensorProto.FLOAT16) if is_float16 else (TensorProto.FLOAT16 if insert_output_cast else TensorProto.FLOAT)
    inputs = [
        helper.make_tensor_value_info(
            "input_0", input_type, ['N', 'N']),
        helper.make_tensor_value_info(
            "input_1", input_type, ['N', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "cast_output_2" if insert_output_cast else ("transpose_output_2" if transpose_output else "product"), output_type, ['N', 'N'])
    ]

    save(model_path + ("_float16"      if is_float16          else "_float" ) +
                      ("_transpose_inputs" if transpose_inputs else "")  +
                      ("_transpose_output" if transpose_output else "")  +
                      ("_input_casts"  if insert_input_casts  else "") +
                      ("_output_cast"  if insert_output_cast else ""),
        nodes, inputs, outputs, [])

for (transpose_inputs, transpose_output, insert_input_casts, insert_output_cast, is_float16) in list(itertools.product([False, True], repeat=5)):
  if insert_input_casts or insert_output_cast:
      gen_propagate_cast_test_model("compute", transpose_inputs, transpose_output, insert_input_casts, insert_output_cast, is_float16)

gen_fuse_sibling_casts("fuse_sibling_casts")
gen_fuse_back2back_casts("fuse_back2back_casts")
