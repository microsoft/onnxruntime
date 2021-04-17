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

def flip_type(flip, type):
    return (TensorProto.FLOAT16 if type == TensorProto.FLOAT else TensorProto.FLOAT) if flip else type
def do_cast_inputs(input_0, input_1, nodes):
    input_cast_type = TensorProto.FLOAT
    nodes.extend([helper.make_node(
        "Cast",
        [input_0],
        ["cast_"+input_0],
        "Cast_0",
        to = input_cast_type),
    helper.make_node(
        "Cast",
        [input_1],
        ["cast_"+input_1],
        "Cast_1",
        to = input_cast_type)])
    return "cast_"+input_0, "cast_"+input_1
def do_transpose_inputs(input_0, input_1, nodes):
    nodes.extend([helper.make_node("Transpose", [input_0], ["transpose_"+input_0], "Transpose_0"),
                    helper.make_node("Transpose", [input_1], ["transpose_"+input_1], "Transpose_1")])
    return "transpose_"+input_0, "transpose_"+input_1
def do_cast_product(product, nodes):
    nodes.append(helper.make_node(
        "Cast",
        [product],
        ["cast" + product],
        "Cast_2",
        to = TensorProto.FLOAT16))
    return "cast_"+product

def gen_propagate_cast_test_model(model_path, transpose_inputs, transpose_product, cast_inputs, cast_product, insert_add, cast_sum, cast_input2):
    nodes = [
        helper.make_node(
            "MatMul",
            ["input_transpose_0" if transpose_inputs else ("cast_input_0" if cast_inputs else "input_0"),
             "input_transpose_1" if transpose_inputs else ("cast_input_1" if cast_inputs else "input_1")],
            ["product"],
            "MatMul_0")
    ]

    if cast_product:
        nodes.append(helper.make_node(
            "Cast",
             ["product_transpose" if transpose_product else "product"],
             ["product_cast"],
             "Cast_2",
             to = TensorProto.FLOAT16))

    if cast_inputs:
        input_cast_type = TensorProto.FLOAT
        nodes.extend([helper.make_node(
            "Cast",
            ["input_0"],
            ["cast_input_0"],
            "Cast_0",
            to = TensorProto.FLOAT),
        helper.make_node(
            "Cast",
            ["input_1"],
            ["cast_input_1"],
            "Cast_1",
            to = TensorProto.FLOAT)])

    if transpose_inputs:
        nodes.extend([helper.make_node("Transpose", ["cast_input_0" if cast_inputs else "input_0"], ["input_transpose_0"], "Transpose_0"),
                      helper.make_node("Transpose", ["cast_input_1" if cast_inputs else "input_1"], ["input_transpose_1"], "Transpose_1")])

    if transpose_product:
        nodes.append(helper.make_node("Transpose", ["product"], ["product_transpose"], "Transpose_2"))

    input_type = TensorProto.FLOAT16 if cast_inputs else TensorProto.FLOAT
    output_type = flip_type(cast_sum, flip_type(cast_product, flip_type(cast_inputs, input_type)))
    inputs = [
        helper.make_tensor_value_info(
            "input_0", input_type, ['N', 'N']),
        helper.make_tensor_value_info(
            "input_1", input_type, ['N', 'N'])
    ]
    if insert_add:
        add_input_type = flip_type(True, input_type) if cast_inputs != cast_product else input_type
        add_input_type = flip_type(cast_input2, add_input_type)
        inputs.append(helper.make_tensor_value_info("input_2", add_input_type, ['N', 'N']))
        nodes.append(helper.make_node("Add", ["product_cast" if cast_product else ("product_transpose" if transpose_product else "product"), "cast_input_2" if cast_input2 else "input_2"], ["sum"], "Add_0"))
        if cast_sum:
            input2_cast_type = flip_type(True, flip_type(cast_input2, add_input_type))
            nodes.append(helper.make_node(
                "Cast",
                ["sum"],
                ["cast_sum"],
                "Cast_3",
                to = input2_cast_type))
        if cast_input2:
            nodes.append(helper.make_node(
                "Cast",
                ["input_2"],
                ["cast_input_2"],
                "Cast_4",
                to = flip_type(True, add_input_type)))
    outputs = [
        helper.make_tensor_value_info(
            "cast_sum" if cast_sum else "sum" if insert_add else ("product_cast" if cast_product else ("product_transpose" if transpose_product else "product")), output_type, ['N', 'N'])
    ]

    save(model_path + ("_transpose_inputs" if transpose_inputs else "")  +
                      ("_transpose_product" if transpose_product else "")  +
                      ("_cast_inputs"  if cast_inputs  else "") +
                      ("_cast_product"  if cast_product else "") +
                      ("_cast_input2"  if cast_input2 else "") +
                      ("_cast_sum"  if cast_sum else ""),
        nodes, inputs, outputs, [])

def gen_matmul_two_products(model_path, transpose, transpose_before_cast, second_matmul):
    def do_transpose(output_0, output_1, nodes):
        nodes.extend([helper.make_node("Transpose", [output_0], ["transpose_0_"+output_0], "Transpose_0"),
            helper.make_node("Transpose", [output_1], ["transpose_1_"+output_1], "Transpose_1")])
        output_0 = "transpose_0_"+output_0
        output_1 ="transpose_1_"+output_1
        return output_0, output_1
    input_type = TensorProto.FLOAT
    input_0 = "input_0"
    input_1 = "input_1"
    output = "product"
    output_0 = "product"
    output_1 = "product"
    inputs = [
        helper.make_tensor_value_info(
            input_0, input_type, ['M', 'K']),
        helper.make_tensor_value_info(
            input_1, input_type, ['K', 'N'])
    ]
    outputs = []
    nodes = [
        helper.make_node(
            "MatMul",
            [input_0, input_1],
            [output],
            "MatMul_0")]
    if second_matmul:
        nodes.append(helper.make_node(
            "MatMul",
            [input_0, input_1],
            ["second_"+output],
            "MatMul_1"))
        outputs.append(helper.make_tensor_value_info(
            "second_"+output,  input_type, ['M', 'N']))

    if transpose and transpose_before_cast:
        output_0, output_1 = do_transpose(output_0, output_1, nodes)

    nodes.append(helper.make_node(
        "Cast",
        [output_0],
        ["cast_0_"+output_0],
        "Cast_0",
        to = TensorProto.FLOAT16))
    output_0 = "cast_0_"+output_0

    if second_matmul:
        nodes.append(helper.make_node(
            "Cast",
            [output_1],
            ["cast_1_"+output_1],
            "Cast_1",
            to = TensorProto.FLOAT16))
        output_1 = "cast_1_"+output_1

    if transpose and not transpose_before_cast:
        output_0, output_1 = do_transpose(output_0, output_1, nodes)

    outputs.extend([
        helper.make_tensor_value_info(
            output_0,  flip_type(True, input_type), ['M', 'N']),
        helper.make_tensor_value_info(
            output_1,  flip_type(second_matmul, input_type)git reset, ['M', 'N'])
    ])
    model_path += ("_transpose_before_cast" if transpose_before_cast else "_transpose_after_cast") if transpose else ""
    model_path +=  "_second_matmul" if second_matmul else ""
    save(model_path, nodes, inputs, outputs, [])

for (transpose_inputs, transpose_product, cast_inputs, cast_product, insert_add, cast_sum, cast_input2) in list(itertools.product([False, True], repeat=7)):
    if not insert_add and (cast_sum or cast_input2):
        continue
    if cast_inputs or cast_product or cast_sum:
        gen_propagate_cast_test_model("matmul_add" if insert_add else "matmul", transpose_inputs, transpose_product, cast_inputs, cast_product, insert_add, cast_sum, cast_input2)

gen_fuse_sibling_casts("fuse_sibling_casts")
gen_fuse_back2back_casts("fuse_back2back_casts")

for (transpose, transpose_before_cast, second_matmul) in list(itertools.product([False, True], repeat=3)):
    if not transpose and transpose_before_cast:
        continue
    gen_matmul_two_products("matmul_two_outputs", transpose, transpose_before_cast, second_matmul)