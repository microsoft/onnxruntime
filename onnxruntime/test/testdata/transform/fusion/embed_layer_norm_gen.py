import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum

def GenerateModel3(model_name):
    nodes = [        # LayerNorm subgraph
        helper.make_node("Shape", ["input_ids"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape1_out", "indices_0"], ["gather0_out"], "gather0"),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Shape", ["input_ids"], ["shape2_out"], "shape2"),
        helper.make_node("Gather", ["shape2_out", "indices_1"], ["gather1_out"], "gather1"),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),
        helper.make_node("Concat", ["unsqueeze0_out", "unsqueeze1_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Cast", ["gather1_out"], ["cast_out"], "cast", to=7),
        helper.make_node("Range", ["start_0", "cast_out", "delta_1"], ["range_out"], "range"),
        helper.make_node("Unsqueeze", ["range_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),
        helper.make_node("Expand", ["unsqueeze2_out", "concat_out"], ["expand_out"], "expand"),
        helper.make_node("Gather", ["pos_embed", "expand_out"], ["pos_gather_out"], "pos_gather"),
        helper.make_node("Gather", ["word_embed", "input_ids"], ["word_gather_out"], "word_gather"),
        helper.make_node("Add", ["word_gather_out", "pos_gather_out"], ["word_add_pos_out"], "word_add_pos"),
        helper.make_node("Gather", ["seg_embed", "segment_ids"], ["seg_gather_out"], "seg_gather"),
        helper.make_node("Add", ["word_add_pos_out", "seg_gather_out"], ["add3_out"], "add3"),
        helper.make_node("LayerNormalization", ["add3_out", "layer_norm_weight", "layer_norm_bias"], ["layernorm_out"], "layernorm", axis=-1, epsion=0.000009999999747378752),
        helper.make_node("Cast", ["input_mask"], ["mask_cast_out"], "mask_cast", to=6),
        helper.make_node("ReduceSum", ["mask_cast_out"], ["mask_index_out"], "mask_index", axes=[1], keepdims=0),
        helper.make_node("Attention", ["layernorm_out", "qkv_weights", "qkv_bias", "mask_index_out"], ["att_out"], "att", domain="com.microsoft", num_heads=2),
        helper.make_node("MatMul", ["att_out", "matmul_weight"], ["matmul_out"], "matmul"),
        helper.make_node("Add", ["matmul_out", "add_bias"], ["add_out"], "add"),
        helper.make_node("Add", ["add_out", "layernorm_out"], ["add2_out"], "add2")
        
    ]

    # hidden_size=4, num_heads=2, max_seq_length=3
    initializers = [ # initializers
        helper.make_tensor('indices_0', TensorProto.INT64, [], [0]),
        helper.make_tensor('indices_1', TensorProto.INT64, [], [1]),
        helper.make_tensor('start_0', TensorProto.INT64, [], [0]),
        helper.make_tensor('delta_1', TensorProto.INT64, [], [1]),
        helper.make_tensor('word_embed', TensorProto.FLOAT, [2, 4], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('pos_embed', TensorProto.FLOAT, [4, 4], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('seg_embed', TensorProto.FLOAT, [2, 4], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('layer_norm_weight', TensorProto.FLOAT, [4], [1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('layer_norm_bias', TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),

        helper.make_tensor('qkv_weights', TensorProto.FLOAT, [4, 4], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('qkv_bias', TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),
        
        helper.make_tensor('matmul_weight', TensorProto.FLOAT, [4, 4], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('add_bias', TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),
    ]

    graph = helper.make_graph(
        nodes,
        "EmbedLayerNorm_format3",  #name
        [  # inputs
            helper.make_tensor_value_info('input_ids', TensorProto.INT64, ['batch', 3]),
            helper.make_tensor_value_info('segment_ids', TensorProto.INT64, ['batch', 3]),
            helper.make_tensor_value_info('input_mask', TensorProto.INT64, ['batch', 3]),
        ],
        [  # outputs
            helper.make_tensor_value_info('add2_out', TensorProto.FLOAT, ['batch', 3, 4]),
        ],
        initializers
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)

def GenerateModel5(model_name):
    batch_size = 2
    hidden_size = 4
    attention_heads = 2
    sequence_length = 3

    nodes = [
        helper.make_node("Gather", ["word_embed", "input_ids"], ["word_gather_out"], "word_gather", axis=0),
        helper.make_node("Add", ["word_gather_out", "pos_gather_out"], ["word_add_pos_out"], "word_add_pos"),
        helper.make_node("Gather", ["seg_embed", "segment_ids"], ["seg_gather_out"], "seg_gather", axis=0),
        helper.make_node("Add", ["word_add_pos_out", "seg_gather_out"], ["add3_out"], "add3"),
        helper.make_node("LayerNormalization", ["add3_out", "layer_norm_weight", "layer_norm_bias"], ["layernorm_out"], "layernorm", axis=-1, epsion=0.000009999999747378752),
        helper.make_node("Cast", ["input_mask"], ["mask_cast_out"], "mask_cast", to=6),
        helper.make_node("ReduceSum", ["mask_cast_out"], ["mask_index_out"], "mask_index", axes=[1], keepdims=0),
        helper.make_node("Attention", ["layernorm_out", "qkv_weights", "qkv_bias", "mask_index_out"], ["att_out"], "att", domain="com.microsoft", num_heads=attention_heads),
        helper.make_node("MatMul", ["att_out", "matmul_weight"], ["matmul_out"], "matmul"),
        helper.make_node("Add", ["matmul_out", "add_bias"], ["add_out"], "add"),
        helper.make_node("Add", ["add_out", "layernorm_out"], ["add2_out"], "add2")
    ]

    qkv_weights = [1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,
                 1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,
                 1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,
                 1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0]

    initializers = [# initializers
        helper.make_tensor('word_embed', TensorProto.FLOAT, [2, hidden_size], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('pos_gather_out', TensorProto.FLOAT, [batch_size, sequence_length, hidden_size],
                           [1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,  9.0, 8.0, 7.0, 6.0,
                            1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,  9.0, 8.0, 7.0, 6.0]),
        helper.make_tensor('seg_embed', TensorProto.FLOAT, [2, hidden_size], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('layer_norm_weight', TensorProto.FLOAT, [hidden_size], [1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('layer_norm_bias', TensorProto.FLOAT, [hidden_size], [0.1, 0.2, 0.3, 0.4]),
        helper.make_tensor('qkv_weights', TensorProto.FLOAT, [hidden_size, 3 * hidden_size], qkv_weights),

        helper.make_tensor('qkv_bias', TensorProto.FLOAT, [3 * hidden_size], [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        
        helper.make_tensor('matmul_weight', TensorProto.FLOAT, [hidden_size, hidden_size], 
                           [1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0,  1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('add_bias', TensorProto.FLOAT, [hidden_size], [0.1, 0.2, 0.3, 0.4]),]

    graph = helper.make_graph(
        nodes,
        "EmbedLayerNorm_format5",  #name
        [  # inputs
            helper.make_tensor_value_info('input_ids', TensorProto.INT64, [batch_size, sequence_length]),
            helper.make_tensor_value_info('segment_ids', TensorProto.INT64, [batch_size, sequence_length]),
            helper.make_tensor_value_info('input_mask', TensorProto.INT64, [batch_size, sequence_length]),
        ],
        [  # outputs
            helper.make_tensor_value_info('add2_out', TensorProto.FLOAT, [batch_size, sequence_length, hidden_size]),
        ],
        initializers
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)

#GenerateModel3('embed_layer_norm_format3.onnx')
GenerateModel5('embed_layer_norm_format5.onnx')