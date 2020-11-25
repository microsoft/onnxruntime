import onnx
from onnx import helper
from onnx import TensorProto

add = helper.make_node("Add", ["input", "bias"], ["add_out"], "add")
reverseadd = helper.make_node("Add", ["bias", "input"], ["add_out"], "add")
softmax1 =  helper.make_node("Softmax", ["add_out"], ["output"], "softmax", axis=1)
softmax3 =  helper.make_node("Softmax", ["add_out"], ["output"], "softmax", axis=3)
softmax6 =  helper.make_node("Softmax", ["add_out"], ["output"], "softmax", axis=6)

onnx.save(
    helper.make_model(
    helper.make_graph(
    [add, softmax1], "Add_Softmax_Fusion",
    [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, ['d_1', 'd_2']),
        helper.make_tensor_value_info('bias', TensorProto.FLOAT, ['d_1', 'd_2']),
    ],
    [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, ['d_1', 'd_2']),
    ], 
    [])), r'bias_softmax_fusion_simple.onnx')

onnx.save(
    helper.make_model(
    helper.make_graph(
    [add, softmax6], "Add_Softmax_Fusion",
    [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
        helper.make_tensor_value_info('bias', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 1, 1, 1, 'd_6', 'd_7', 'd_8']),
    ],
    [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
    ], 
    [])), r'bias_softmax_fusion_middleones.onnx')

onnx.save(
    helper.make_model(
    helper.make_graph(
    [reverseadd, softmax6], "Add_Softmax_Fusion",
    [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
        helper.make_tensor_value_info('bias', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 1, 1, 1, 'd_6', 'd_7', 'd_8']),
    ],
    [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
    ], 
    [])), r'bias_softmax_fusion_middleones_reversed.onnx')

# should NOT fuse
onnx.save(
    helper.make_model(
    helper.make_graph(
    [add, softmax3], "Add_Softmax_Fusion",
    [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
        helper.make_tensor_value_info('bias', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 1, 1, 1, 'd_6', 'd_7', 'd_8']),
    ],
    [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
    ], 
    [])), r'bias_softmax_fusion_middleones_badaxis.onnx')

onnx.save(
    helper.make_model(
    helper.make_graph(
    [add, softmax6], "Add_Softmax_Fusion",
    [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
        helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 1, 1, 1, 1, 1, 'd_6', 'd_7', 'd_8']),
    ],
    [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
    ], 
    [])), r'bias_softmax_fusion_allleadingones.onnx')

onnx.save(
    helper.make_model(
    helper.make_graph(
    [add, softmax6], "Add_Softmax_Fusion",
    [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
        helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 1, 'd_6', 'd_7', 'd_8']),
    ],
    [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
    ], 
    [])), r'bias_softmax_fusion_someleadingones.onnx')

onnx.save(
    helper.make_model(
    helper.make_graph(
    [add, softmax6], "Add_Softmax_Fusion",
    [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
        helper.make_tensor_value_info('bias', TensorProto.FLOAT, ['d_6', 'd_7', 'd_8']),
    ],
    [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8']),
    ], 
    [])), r'bias_softmax_fusion_noleadingones.onnx')