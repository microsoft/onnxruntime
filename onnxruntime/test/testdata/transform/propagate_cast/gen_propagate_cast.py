import itertools

import numpy as np
import onnx
from onnx import OperatorSetIdProto, TensorProto, helper

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
    return "float" if type == TensorProto.FLOAT else "float16"


def save(model_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, "CastPropagateTest", inputs, outputs, initializers)

    model = helper.make_model(graph, opset_imports=opsets, producer_name="onnxruntime-test")

    onnx.save(model, model_path + ".onnx")


def gen_fuse_back2back_casts(model_path):

    for (type1, type2) in list(itertools.product([TensorProto.FLOAT, TensorProto.FLOAT16], repeat=2)):

        nodes = [
            helper.make_node("MatMul", ["input_0", "input_1"], ["product"], "MatMul_0"),
            helper.make_node("Cast", ["product"], ["product_cast"], "Cast_0", to=type1),
            helper.make_node("Cast", ["product_cast"], ["output"], "Cast_1", to=type2),
        ]
        input_type = (
            type2 if type1 != type2 else (TensorProto.FLOAT16 if type1 == TensorProto.FLOAT else TensorProto.FLOAT)
        )
        output_type = (
            input_type
            if type1 != type2
            else (TensorProto.FLOAT16 if input_type == TensorProto.FLOAT else TensorProto.FLOAT)
        )
        inputs = [
            helper.make_tensor_value_info("input_0", input_type, ["M", "K"]),
            helper.make_tensor_value_info("input_1", input_type, ["K", "N"]),
        ]

        outputs = [
            helper.make_tensor_value_info("output", output_type, ["M", "N"]),
        ]

        save(
            model_path + "_" + type_to_string(type1) + "_" + type_to_string(type2),
            nodes,
            inputs,
            outputs,
            [],
        )


def gen_fuse_sibling_casts(model_path):

    for (type1, type2) in list(itertools.product([TensorProto.FLOAT, TensorProto.FLOAT16], repeat=2)):
        input_type = (
            type2 if type1 != type2 else (TensorProto.FLOAT16 if type1 == TensorProto.FLOAT else TensorProto.FLOAT)
        )
        nodes = [
            helper.make_node("MatMul", ["input_0", "input_1"], ["product"], "MatMul_0"),
            helper.make_node("Cast", ["product"], ["cast_0_output"], "Cast_0", to=type1),
            helper.make_node("Identity", ["cast_0_output"], ["output_0"], "Identity_0"),
            helper.make_node("Cast", ["product"], ["cast_1_output"], "Cast_1", to=type2),
            helper.make_node("Identity", ["cast_1_output"], ["output_1"], "Identity_1"),
        ]

        inputs = [
            helper.make_tensor_value_info("input_0", input_type, ["M", "K"]),
            helper.make_tensor_value_info("input_1", input_type, ["K", "N"]),
        ]

        outputs = [
            helper.make_tensor_value_info("output_0", type1, ["M", "N"]),
            helper.make_tensor_value_info("output_1", type2, ["M", "N"]),
        ]

        save(
            model_path + "_" + type_to_string(type1) + "_" + type_to_string(type2),
            nodes,
            inputs,
            outputs,
            [],
        )


def flip_type(type, flip=True):
    return (TensorProto.FLOAT16 if type == TensorProto.FLOAT else TensorProto.FLOAT) if flip else type


def do_cast_inputs(input_0, input_1, nodes, input_cast_type):
    nodes.extend(
        [
            helper.make_node("Cast", [input_0], ["cast_" + input_0], "Cast_0", to=input_cast_type),
            helper.make_node("Cast", [input_1], ["cast_" + input_1], "Cast_1", to=input_cast_type),
        ]
    )
    return "cast_" + input_0, "cast_" + input_1


def do_transpose_inputs(input_0, input_1, nodes):
    nodes.extend(
        [
            helper.make_node("Transpose", [input_0], ["input_transpose_0"], "Transpose_0"),
            helper.make_node("Transpose", [input_1], ["input_transpose_1"], "Transpose_1"),
        ]
    )
    return "input_transpose_0", "input_transpose_1"


def do_cast_product(product, nodes, product_type):
    nodes.insert(
        1,
        helper.make_node("Cast", [product], [product + "_cast"], "Cast_2", to=product_type),
    )
    return product + "_cast"


def do_transpose_product(product, nodes):
    if transpose_product:
        nodes.append(helper.make_node("Transpose", [product], [product + "_transpose"], "Transpose_2"))
    return product + "_transpose"


def do_cast_sum(sum, nodes, type):
    nodes.append(helper.make_node("Cast", [sum], ["cast_" + sum], "Cast_3", to=type))
    return "cast_" + sum


def do_cast_input2(input_2, nodes, type):
    nodes.append(helper.make_node("Cast", [input_2], ["cast_" + input_2], "Cast_4", to=type))
    return "cast_" + input_2


def gen_propagate_cast_test_model(
    model_path,
    transpose_inputs,
    transpose_product,
    cast_inputs,
    cast_product,
    insert_add,
    cast_sum,
    cast_input2,
    transpose_inputs_before_cast=False,
):
    input_0 = "input_0"
    input_1 = "input_1"
    product = "product"
    nodes = []
    input_type = TensorProto.FLOAT
    product_type = input_type
    if transpose_inputs_before_cast:
        if transpose_inputs:
            input_0, input_1 = do_transpose_inputs(input_0, input_1, nodes)
        if cast_inputs:
            input_0, input_1 = do_cast_inputs(input_0, input_1, nodes, input_type)
            input_type = flip_type(input_type)
    else:
        if cast_inputs:
            input_0, input_1 = do_cast_inputs(input_0, input_1, nodes, input_type)
            input_type = flip_type(input_type)
        if transpose_inputs:
            input_0, input_1 = do_transpose_inputs(input_0, input_1, nodes)
    nodes.append(helper.make_node("MatMul", [input_0, input_1], [product], "MatMul_0"))
    if transpose_product:
        product = do_transpose_product(product, nodes)

    if cast_product:
        product = do_cast_product(product, nodes, flip_type(product_type))
        product_type = flip_type(product_type)

    inputs = [
        helper.make_tensor_value_info("input_0", input_type, ["N", "N"]),
        helper.make_tensor_value_info("input_1", input_type, ["N", "N"]),
    ]
    if insert_add:
        input_2 = "input_2"
        add_input_type = flip_type(product_type, cast_input2)
        inputs.append(helper.make_tensor_value_info(input_2, add_input_type, ["N", "N"]))
        output = "sum"
        output_type = product_type
        if cast_input2:
            input_2 = do_cast_input2(input_2, nodes, flip_type(add_input_type))
        nodes.append(helper.make_node("Add", [product, input_2], [output], "Add_0"))
        if cast_sum:
            output = do_cast_sum(output, nodes, flip_type(output_type))
            output_type = flip_type(output_type)
    else:
        output = product
        output_type = product_type
    outputs = [helper.make_tensor_value_info(output, output_type, ["N", "N"])]

    save(
        model_path
        + ("_transpose_inputs" if transpose_inputs else "")
        + ("_transpose_product" if transpose_product else "")
        + ("_cast_inputs" if cast_inputs else "")
        + ("_cast_product" if cast_product else "")
        + ("_cast_input2" if cast_input2 else "")
        + ("_cast_sum" if cast_sum else ""),
        nodes,
        inputs,
        outputs,
        [],
    )


def gen_matmul_two_products(model_path, transpose, transpose_before_cast, second_matmul, cast_inputs):
    def do_transpose(output_0, output_1, transpose, nodes):
        nodes.append(helper.make_node("Transpose", [output_0], ["transpose_0_" + output_0], "Transpose_0"))
        output_0 = "transpose_0_" + output_0
        if transpose > 1:
            nodes.append(helper.make_node("Transpose", [output_1], ["transpose_1_" + output_1], "Transpose_1"))
            output_1 = "transpose_1_" + output_1
        return output_0, output_1

    input_type = flip_type(TensorProto.FLOAT, cast_inputs)
    input_0 = "input_0"
    input_1 = "input_1"
    output = "product"
    output_0 = "product"
    output_1 = "product"
    outputs = []
    nodes = []
    cast_count = 0
    inputs = [
        helper.make_tensor_value_info("input_0", input_type, ["M", "K"]),
        helper.make_tensor_value_info("input_1", input_type, ["K", "N"]),
    ]
    if cast_inputs:
        input_type = flip_type(input_type)
        input_0, input_1 = do_cast_inputs(input_0, input_1, nodes, input_type)
        cast_count += 2
    output0_type = input_type
    output1_type = input_type
    nodes.append(helper.make_node("MatMul", [input_0, input_1], [output], "MatMul_0"))
    if second_matmul:
        nodes.append(helper.make_node("MatMul", [input_0, input_1], ["second_" + output], "MatMul_1"))
        outputs.append(helper.make_tensor_value_info("second_" + output, input_type, ["M", "N"]))
        if add_products:
            nodes.append(helper.make_node("Add", [output, "second_" + output], ["sum"], "Add_0"))
            outputs.append(helper.make_tensor_value_info("sum", input_type, ["M", "N"]))
    if transpose > 0 and transpose_before_cast:
        output_0, output_1 = do_transpose(output_0, output_1, transpose, nodes)
    output0_type = flip_type(output0_type)
    nodes.append(
        helper.make_node(
            "Cast",
            [output_0],
            ["cast_" + str(cast_count) + "_" + output_0],
            "Cast_" + str(cast_count),
            to=output0_type,
        )
    )
    output_0 = "cast_" + str(cast_count) + "_" + output_0
    cast_count += 1
    if second_matmul:
        nodes.append(
            helper.make_node(
                "Cast",
                [output_1],
                ["cast_" + str(cast_count) + "_" + output_1],
                "Cast_" + str(cast_count),
                to=TensorProto.FLOAT16,
            )
        )
        output_1 = "cast_" + str(cast_count) + "_" + output_1
        output1_type = flip_type(output1_type)
    if transpose > 0 and not transpose_before_cast:
        output_0, output_1 = do_transpose(output_0, output_1, transpose, nodes)

    outputs.extend(
        [
            helper.make_tensor_value_info(output_0, output0_type, ["M", "N"]),
            helper.make_tensor_value_info(output_1, output1_type, ["M", "N"]),
        ]
    )
    model_path += "_cast_inputs" if cast_inputs else ""
    model_path += (
        ("_transpose_before_cast" if transpose_before_cast else "_transpose_after_cast") if transpose > 0 else ""
    )
    model_path += "_transpose" if transpose > 1 else ""
    model_path += "_second_matmul" if second_matmul else ""
    model_path += "_add_products" if add_products else ""
    save(model_path, nodes, inputs, outputs, [])


def gen_bool_to_float16_cast(model_path):
    X1 = helper.make_tensor_value_info("x1", TensorProto.INT64, [1, 1])
    X2 = helper.make_tensor_value_info("x2", TensorProto.INT64, [1, 1])
    X3 = helper.make_tensor_value_info("x3", TensorProto.FLOAT, [1, 1])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 1])

    less1 = helper.make_node("Less", ["x1", "x2"], ["less1"], name="less1")
    cast1 = helper.make_node("Cast", ["less1"], ["cast1"], name="cast1", to=TensorProto.FLOAT16)
    cast2 = helper.make_node("Cast", ["x3"], ["cast2"], name="cast2", to=TensorProto.FLOAT16)
    add1 = helper.make_node("Add", ["cast1", "cast2"], ["output"])

    save(model_path, [less1, cast1, cast2, add1], [X1, X2, X3], [Y], [])


def gen_bool_to_float_cast(model_path):
    X1 = helper.make_tensor_value_info("x1", TensorProto.INT64, [1, 1])
    X2 = helper.make_tensor_value_info("x2", TensorProto.INT64, [1, 1])
    X3 = helper.make_tensor_value_info("x3", TensorProto.FLOAT16, [1, 1])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 1])

    less1 = helper.make_node("Less", ["x1", "x2"], ["less1"], name="less1")
    cast1 = helper.make_node("Cast", ["less1"], ["cast1"], name="cast1", to=TensorProto.FLOAT)
    cast2 = helper.make_node("Cast", ["x3"], ["cast2"], name="cast2", to=TensorProto.FLOAT)
    add1 = helper.make_node("Add", ["cast1", "cast2"], ["add1"])
    cast3 = helper.make_node("Cast", ["add1"], ["output"], name="cast3", to=TensorProto.FLOAT16)

    save(model_path, [less1, cast1, cast2, cast3, add1], [X1, X2, X3], [Y], [])


def gen_one_input_one_output_test(op, model_path, axes_attribute=False):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 2])
    output_shape = [2, 2]
    if op == "Unsqueeze":
        output_shape.append(1)
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, output_shape)
    node_inputs = []
    graph_inputs = [X]
    cast1 = helper.make_node("Cast", ["x"], ["cast1"], name="cast1", to=TensorProto.FLOAT)
    node_inputs.insert(0, "cast1")
    if axes_attribute:
        node = helper.make_node(
            op,
            node_inputs,
            ["op_output"],
            name=op + str(1),
            axes=np.array([2]).astype(np.int64),
        )
    else:
        node = helper.make_node(op, node_inputs, ["op_output"], name=op + str(1))
    cast2 = helper.make_node("Cast", ["op_output"], ["y"], name="cast2", to=TensorProto.FLOAT16)
    save(model_path, [cast1, node, cast2], graph_inputs, [Y], [])


for (
    transpose_inputs,
    transpose_product,
    cast_inputs,
    cast_product,
    insert_add,
    cast_sum,
    cast_input2,
) in list(itertools.product([False, True], repeat=7)):
    if not insert_add and (cast_sum or cast_input2):
        continue
    if cast_inputs or cast_product or cast_sum:
        gen_propagate_cast_test_model(
            "matmul_add" if insert_add else "matmul",
            transpose_inputs,
            transpose_product,
            cast_inputs,
            cast_product,
            insert_add,
            cast_sum,
            cast_input2,
        )

gen_fuse_sibling_casts("fuse_sibling_casts")
gen_fuse_back2back_casts("fuse_back2back_casts")

for (
    transpose,
    transpose_before_cast,
    second_matmul,
    add_products,
    cast_inputs,
) in list(itertools.product([0, 1, 2], [False, True], [False, True], [False, True], [False, True])):
    if not transpose and transpose_before_cast:
        continue
    if not second_matmul and add_products:
        continue
    gen_matmul_two_products(
        "matmul_two_outputs",
        transpose,
        transpose_before_cast,
        second_matmul,
        cast_inputs,
    )


gen_bool_to_float16_cast("negative_test_case_bool_fp16_cast")
gen_bool_to_float_cast("negative_test_case_bool_fp_cast")
gen_one_input_one_output_test("Squeeze", "squeeze_cast_propagation_test")
gen_one_input_one_output_test("Unsqueeze", "unsqueeze_cast_propagation_test", True)
