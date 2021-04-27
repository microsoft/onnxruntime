from onnx import load_model, helper, external_data_helper, save_model, TensorProto
#onnx_model_path = "/bert_ort/wy/Megatron/10B_ChitChat_eot_onnx_split_op14/ChitChat_fp16_op14.onnx"
onnx_model_path = "/bert_ort/wy/Megatron/10B_ChitChat_eot_onnx_split_op14_optimized/ChitChat_fp16_op14.onnx"
#onnx_model_path = "/bert_ort/wy/Megatron/ChitChatONNX/megatron_onnx_partial_6_layer/fp16_split.onnx"
#fixed_onnx_model_path = "/bert_ort/wy/Transformers/megatron/onnxruntime/python/tools/transformers/megatron_optimized_fixed/fp16_merge_optimized.onnx"
fixed_onnx_model_path = "/bert_ort/wy/Megatron/10B_ChitChat_eot_onnx_split_op14_optimized_fixed/ChitChat_fp16_op14.onnx"

model = load_model(onnx_model_path, format = None, load_external_data = True)
model.opset_import[0].version = 14

nodes_to_remove = []
nodes_to_add = []
split_v_1 = [4096, 4096, 4096]
init_split_1 = helper.make_tensor('split_v_1', TensorProto.INT64, [3], split_v_1)
split_v_2 = [1, 1]
init_split_2 = helper.make_tensor('split_v_2', TensorProto.INT64, [2], split_v_2)
model.graph.initializer.extend([init_split_1, init_split_2])
for node in model.graph.node:
    if 'Squeeze' in node.name:
        init_axes = helper.make_tensor('axes' + node.name, TensorProto.INT64, [1], node.attribute[0].ints)
        model.graph.initializer.extend([init_axes])
        new_squeeze_node = helper.make_node(
            "Squeeze",
            inputs = [node.input[0], 'axes' + node.name],
            outputs = node.output,
            name = node.name)
        nodes_to_add.extend([new_squeeze_node])
        nodes_to_remove.extend([node])
        print("replace " + node.name)

    if 'Split' in node.name:
        new_split_node = None
        if len(node.output) == 2:
            new_split_node = helper.make_node(
                'Split',
                inputs = [node.input[0], 'split_v_2'],
                outputs = node.output,
                name = node.name,
                axis=0,
            )
        else:
            new_split_node = helper.make_node(
                'Split',
                inputs = [node.input[0], 'split_v_1'],
                outputs = node.output,
                name = node.name,
                axis=2,
            )
        nodes_to_add.extend([new_split_node])
        nodes_to_remove.extend([node])
        print("replace " + node.name)

    if 'Unsqueeze' in node.name:
        init_axes = helper.make_tensor('axes' + node.name, TensorProto.INT64, [1], node.attribute[0].ints)
        model.graph.initializer.extend([init_axes])
        new_unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs = [node.input[0], 'axes' + node.name],
            outputs = node.output,
            name = node.name)
        nodes_to_add.extend([new_unsqueeze_node])
        nodes_to_remove.extend([node])
        print("replace " + node.name)

model.graph.node.extend(nodes_to_add)
for node_to_remove in nodes_to_remove:
    model.graph.node.remove(node_to_remove)

external_data_helper.convert_model_to_external_data(model,
                                                    all_tensors_to_one_file=False,
                                                    location=fixed_onnx_model_path)
save_model(model, fixed_onnx_model_path)


print("try create inference session")
import os
os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = str(0)
import onnxruntime as ort

sess_options = ort.SessionOptions()
ort_session = ort.InferenceSession(fixed_onnx_model_path, sess_options)

print("session created")