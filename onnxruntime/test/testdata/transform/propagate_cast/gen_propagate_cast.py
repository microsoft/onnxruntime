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



def save(model_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(
        nodes,
        "CastPropagateTest",
        inputs, outputs, initializers)

    model = helper.make_model(
        graph, opset_imports=opsets, producer_name="onnxruntime-test")

    onnx.save(model, model_path)

def gen_fuse_back2back_casts(model_path):
    i=0
    l1 = list(itertools.permutations([TensorProto.FLOAT, TensorProto.FLOAT16]))
    l2 = list(itertools.combinations_with_replacement([TensorProto.FLOAT, TensorProto.FLOAT16],2))

    l1.extend(x for x in l2 if x not in l1)
    for (type1, type2) in l1:

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

        save(model_path + "_" + str(i) + ".onnx", nodes, inputs, outputs, [])
        i += 1

def gen_fuse_sibling_casts(model_path):
    i=0
    l1 = list(itertools.permutations([TensorProto.FLOAT, TensorProto.FLOAT16]))
    l2 = list(itertools.combinations_with_replacement([TensorProto.FLOAT, TensorProto.FLOAT16],2))

    l1.extend(x for x in l2 if x not in l1)
    for (type1, type2) in l1:
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
                ["output_0"],
                "Cast_0",
                to = type1),
            helper.make_node(
                "Cast",
                ["product"],
                ["output_1"],
                "Cast_1",
                to = type2)
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

        save(model_path + "_" + str(i) + ".onnx", nodes, inputs, outputs, [])
        i += 1

def gen_propagate_cast_float16(model_path):
    nodes = [
        helper.make_node(
            "MatMul",
            ["input_0", "input_1"],
            ["product"],
            "MatMul_0"),
        helper.make_node(
            "Cast",
            ["product"],
            ["output"],
            "Cast_0",
            to = TensorProto.FLOAT16)
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT, ['M', 'K']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT, ['K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT16, ['M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, [])

def gen_propagate_cast_float(model_path):
    nodes = [
        helper.make_node(
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
            to = TensorProto.FLOAT),
        helper.make_node(
            "MatMul",
            ["cast_input_0", "cast_input_1"],
            ["product"],
            "MatMul_0"),
        helper.make_node(
            "Cast",
            ["product"],
            ["output"],
            "Cast_2",
            to = TensorProto.FLOAT16),
     ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT16, ['M', 'K']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT16, ['K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT16, ['M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, [])

def gen_propagate_cast_float_0(model_path):
    nodes = [
        helper.make_node(
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
            to = TensorProto.FLOAT),
        helper.make_node(
            "MatMul",
            ["cast_input_0", "cast_input_1"],
            ["product"],
            "MatMul_0"),
        helper.make_node(
            "Cast",
            ["product"],
            ["output"],
            "Cast_2",
            to = TensorProto.FLOAT16),
     ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT16, ['M', 'K']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT16, ['K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT16, ['M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, [])

def gen_propagate_cast_float_1(model_path):
    nodes = [
        helper.make_node(
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
            to = TensorProto.FLOAT),
        helper.make_node(
            "MatMul",
            ["cast_input_0", "cast_input_1"],
            ["output"],
            "MatMul_0"),
     ]

    inputs = [
        helper.make_tensor_value_info(
            "input_0", TensorProto.FLOAT16, ['M', 'K']),
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT16, ['K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ['M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, [])
gen_propagate_cast_float16("propagate_cast_float16.onnx")
gen_propagate_cast_float_0("propagate_cast_float_0.onnx")
gen_propagate_cast_float_1("propagate_cast_float_1.onnx")
gen_fuse_sibling_casts("fuse_sibling_casts")
gen_fuse_back2back_casts("fuse_back2back_casts")
