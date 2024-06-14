from enum import Enum  # noqa: F401

import onnx
from onnx import TensorProto, helper
from packaging import version

if version.parse(onnx.__version__) == version.parse("1.8.0"):
    opset_version = 13
elif version.parse(onnx.__version__) == version.parse("1.6.0"):
    opset_version = 11
else:
    raise RuntimeError("Please pip install onnx==1.8.0 or 1.6.0 before running this script")


def GenerateNodes(model_name, has_cast, suffix=""):  # noqa: N802
    nodes = [  # LayerNorm subgraph
        helper.make_node("Shape", ["input_ids" + suffix], ["shape1_out" + suffix], "shape1" + suffix),
        helper.make_node(
            "Gather",
            ["shape1_out" + suffix, "indices_0"],
            ["gather0_out" + suffix],
            "gather0" + suffix,
        ),
        (
            helper.make_node(
                "Unsqueeze",
                ["gather0_out" + suffix, "axes_0"],
                ["unsqueeze0_out" + suffix],
                "unsqueeze0" + suffix,
            )
            if opset_version == 13
            else helper.make_node(
                "Unsqueeze",
                ["gather0_out" + suffix],
                ["unsqueeze0_out" + suffix],
                "unsqueeze0" + suffix,
                axes=[0],
            )
        ),
        helper.make_node("Shape", ["input_ids" + suffix], ["shape2_out" + suffix], "shape2" + suffix),
        helper.make_node(
            "Gather",
            ["shape2_out" + suffix, "indices_1"],
            ["gather1_out" + suffix],
            "gather1" + suffix,
        ),
        (
            helper.make_node(
                "Unsqueeze",
                ["gather1_out" + suffix, "axes_0"],
                ["unsqueeze1_out" + suffix],
                "unsqueeze1" + suffix,
            )
            if opset_version == 13
            else helper.make_node(
                "Unsqueeze",
                ["gather1_out" + suffix],
                ["unsqueeze1_out" + suffix],
                "unsqueeze1" + suffix,
                axes=[0],
            )
        ),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out" + suffix, "unsqueeze1_out" + suffix],
            ["concat_out" + suffix],
            "concat" + suffix,
            axis=0,
        ),
        helper.make_node(
            "Cast",
            ["gather1_out" + suffix],
            ["cast_out" + suffix],
            "cast" + suffix,
            to=7,
        ),
        helper.make_node(
            "Range",
            [
                "start_0",
                "cast_out" + suffix if has_cast else "gather1_out" + suffix,
                "delta_1",
            ],
            ["range_out" + suffix],
            "range" + suffix,
        ),
        (
            helper.make_node(
                "Unsqueeze",
                ["range_out" + suffix, "axes_0"],
                ["unsqueeze2_out" + suffix],
                "unsqueeze2" + suffix,
            )
            if opset_version == 13
            else helper.make_node(
                "Unsqueeze",
                ["range_out" + suffix],
                ["unsqueeze2_out" + suffix],
                "unsqueeze2" + suffix,
                axes=[0],
            )
        ),
        helper.make_node(
            "Expand",
            ["unsqueeze2_out" + suffix, "concat_out" + suffix],
            ["expand_out" + suffix],
            "expand" + suffix,
        ),
        helper.make_node(
            "Gather",
            ["pos_embed", "expand_out" + suffix],
            ["pos_gather_out" + suffix],
            "pos_gather" + suffix,
        ),
        helper.make_node(
            "Gather",
            ["word_embed", "input_ids" + suffix],
            ["word_gather_out" + suffix],
            "word_gather" + suffix,
        ),
        helper.make_node(
            "Add",
            ["word_gather_out" + suffix, "pos_gather_out" + suffix],
            ["word_add_pos_out" + suffix],
            "word_add_pos" + suffix,
        ),
        helper.make_node(
            "Gather",
            ["seg_embed", "segment_ids" + suffix],
            ["seg_gather_out" + suffix],
            "seg_gather" + suffix,
        ),
        helper.make_node(
            "Add",
            ["word_add_pos_out" + suffix, "seg_gather_out" + suffix],
            ["add3_out" + suffix],
            "add3" + suffix,
        ),
        helper.make_node(
            "LayerNormalization",
            ["add3_out" + suffix, "layer_norm_weight", "layer_norm_bias"],
            ["layernorm_out" + suffix],
            "layernorm" + suffix,
            axis=-1,
            epsion=0.000009999999747378752,
        ),
        helper.make_node(
            "Cast",
            ["input_mask" + suffix],
            ["mask_cast_out" + suffix],
            "mask_cast" + suffix,
            to=6,
        ),
        (
            helper.make_node(
                "ReduceSum",
                ["mask_cast_out" + suffix, "axes_1"],
                ["mask_index_out" + suffix],
                "mask_index" + suffix,
                keepdims=0,
            )
            if opset_version == 13
            else helper.make_node(
                "ReduceSum",
                ["mask_cast_out" + suffix],
                ["mask_index_out" + suffix],
                "mask_index" + suffix,
                axes=[1],
                keepdims=0,
            )
        ),
        helper.make_node(
            "Attention",
            [
                "layernorm_out" + suffix,
                "qkv_weights",
                "qkv_bias",
                "mask_index_out" + suffix,
            ],
            ["att_out" + suffix],
            "att" + suffix,
            domain="com.microsoft",
            num_heads=2,
        ),
        helper.make_node(
            "MatMul",
            ["att_out" + suffix, "matmul_weight"],
            ["matmul_out" + suffix],
            "matmul" + suffix,
        ),
        helper.make_node(
            "Add",
            ["matmul_out" + suffix, "add_bias"],
            ["add_out" + suffix],
            "add" + suffix,
        ),
        helper.make_node(
            "Add",
            ["add_out" + suffix, "layernorm_out" + suffix],
            ["add2_out" + suffix],
            "add2" + suffix,
        ),
    ]

    if not has_cast:
        del nodes[7:8]
    return nodes


def GenerateInitializers():  # noqa: N802
    # hidden_size=4, num_heads=2
    initializers = [  # initializers
        helper.make_tensor("indices_0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices_1", TensorProto.INT64, [], [1]),
        helper.make_tensor("start_0", TensorProto.INT64, [], [0]),
        helper.make_tensor("delta_1", TensorProto.INT64, [], [1]),
        helper.make_tensor(
            "word_embed",
            TensorProto.FLOAT,
            [2, 4],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        helper.make_tensor(
            "pos_embed",
            TensorProto.FLOAT,
            [4, 4],
            [
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        ),
        helper.make_tensor(
            "seg_embed",
            TensorProto.FLOAT,
            [2, 4],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        helper.make_tensor("layer_norm_weight", TensorProto.FLOAT, [4], [1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor("layer_norm_bias", TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),
        helper.make_tensor("qkv_weights", TensorProto.FLOAT, [4, 12], [0.1] * 4 * 12),
        helper.make_tensor(
            "qkv_bias",
            TensorProto.FLOAT,
            [12],
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        ),
        helper.make_tensor(
            "matmul_weight",
            TensorProto.FLOAT,
            [4, 4],
            [
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        ),
        helper.make_tensor("add_bias", TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),
        helper.make_tensor("axes_0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("axes_1", TensorProto.INT64, [1], [1]),
    ]

    return initializers


def GenerateMultipleEmbedModel(model_name):  # noqa: N802
    nodes_1 = GenerateNodes(model_name, False, "_1")
    nodes_2 = GenerateNodes(model_name, False, "_2")
    nodes = nodes_1 + nodes_2
    nodes.append(helper.make_node("Add", ["add2_out_1", "add2_out_2"], ["add3_out"], "add3"))

    # hidden_size=4, num_heads=2, max_seq_length=3
    initializers = GenerateInitializers()

    graph = helper.make_graph(
        nodes,
        "EmbedLayerNorm_format3",  # name
        [  # inputs
            helper.make_tensor_value_info("input_ids_1", TensorProto.INT64, ["batch", 3]),
            helper.make_tensor_value_info("segment_ids_1", TensorProto.INT64, ["batch", 3]),
            helper.make_tensor_value_info("input_mask_1", TensorProto.INT64, ["batch", 3]),
            helper.make_tensor_value_info("input_ids_2", TensorProto.INT64, ["batch", 3]),
            helper.make_tensor_value_info("segment_ids_2", TensorProto.INT64, ["batch", 3]),
            helper.make_tensor_value_info("input_mask_2", TensorProto.INT64, ["batch", 3]),
        ],
        [  # outputs
            helper.make_tensor_value_info("add3_out", TensorProto.FLOAT, ["batch", 3, 4]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


def GenerateModel3(model_name, has_cast):  # noqa: N802
    nodes = GenerateNodes(model_name, has_cast)

    # hidden_size=4, num_heads=2, max_seq_length=3
    initializers = GenerateInitializers()

    graph = helper.make_graph(
        nodes,
        "EmbedLayerNorm_format3",  # name
        [  # inputs
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", 3]),
            helper.make_tensor_value_info("segment_ids", TensorProto.INT64, ["batch", 3]),
            helper.make_tensor_value_info("input_mask", TensorProto.INT64, ["batch", 3]),
        ],
        [  # outputs
            helper.make_tensor_value_info("add2_out", TensorProto.FLOAT, ["batch", 3, 4]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


def GenerateModel5(model_name):  # noqa: N802
    batch_size = 2
    hidden_size = 4
    attention_heads = 2
    sequence_length = 3

    nodes = [
        helper.make_node(
            "Gather",
            ["word_embed", "input_ids"],
            ["word_gather_out"],
            "word_gather",
            axis=0,
        ),
        helper.make_node(
            "Add",
            ["word_gather_out", "pos_gather_out"],
            ["word_add_pos_out"],
            "word_add_pos",
        ),
        helper.make_node(
            "Gather",
            ["seg_embed", "segment_ids"],
            ["seg_gather_out"],
            "seg_gather",
            axis=0,
        ),
        helper.make_node("Add", ["word_add_pos_out", "seg_gather_out"], ["add3_out"], "add3"),
        helper.make_node(
            "LayerNormalization",
            ["add3_out", "layer_norm_weight", "layer_norm_bias"],
            ["layernorm_out"],
            "layernorm",
            axis=-1,
            epsion=0.000009999999747378752,
        ),
        helper.make_node("Cast", ["input_mask"], ["mask_cast_out"], "mask_cast", to=6),
        (
            helper.make_node(
                "ReduceSum",
                ["mask_cast_out", "axes_1"],
                ["mask_index_out"],
                "mask_index",
                keepdims=0,
            )
            if opset_version == 13
            else helper.make_node(
                "ReduceSum",
                ["mask_cast_out"],
                ["mask_index_out"],
                "mask_index",
                axes=[1],
                keepdims=0,
            )
        ),
        helper.make_node(
            "Attention",
            ["layernorm_out", "qkv_weights", "qkv_bias", "mask_index_out"],
            ["att_out"],
            "att",
            domain="com.microsoft",
            num_heads=attention_heads,
        ),
        helper.make_node("MatMul", ["att_out", "matmul_weight"], ["matmul_out"], "matmul"),
        helper.make_node("Add", ["matmul_out", "add_bias"], ["add_out"], "add"),
        helper.make_node("Add", ["add_out", "layernorm_out"], ["add2_out"], "add2"),
    ]

    qkv_weights = [1.0] * hidden_size * (3 * hidden_size)

    initializers = [  # initializers
        helper.make_tensor(
            "word_embed",
            TensorProto.FLOAT,
            [2, hidden_size],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        helper.make_tensor(
            "pos_gather_out",
            TensorProto.FLOAT,
            [batch_size, sequence_length, hidden_size],
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                8.0,
                7.0,
                6.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                8.0,
                7.0,
                6.0,
            ],
        ),
        helper.make_tensor(
            "seg_embed",
            TensorProto.FLOAT,
            [2, hidden_size],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        helper.make_tensor("layer_norm_weight", TensorProto.FLOAT, [hidden_size], [1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor("layer_norm_bias", TensorProto.FLOAT, [hidden_size], [0.1, 0.2, 0.3, 0.4]),
        helper.make_tensor(
            "qkv_weights",
            TensorProto.FLOAT,
            [hidden_size, 3 * hidden_size],
            qkv_weights,
        ),
        helper.make_tensor(
            "qkv_bias",
            TensorProto.FLOAT,
            [3 * hidden_size],
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        ),
        helper.make_tensor(
            "matmul_weight",
            TensorProto.FLOAT,
            [hidden_size, hidden_size],
            [
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        ),
        helper.make_tensor("add_bias", TensorProto.FLOAT, [hidden_size], [0.1, 0.2, 0.3, 0.4]),
        helper.make_tensor("axes_1", TensorProto.INT64, [1], [1]),
    ]

    graph = helper.make_graph(
        nodes,
        "EmbedLayerNorm_format5",  # name
        [  # inputs
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [batch_size, sequence_length]),
            helper.make_tensor_value_info("segment_ids", TensorProto.INT64, [batch_size, sequence_length]),
            helper.make_tensor_value_info("input_mask", TensorProto.INT64, [batch_size, sequence_length]),
        ],
        [  # outputs
            helper.make_tensor_value_info(
                "add2_out",
                TensorProto.FLOAT,
                [batch_size, sequence_length, hidden_size],
            ),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


def GenerateModel6(model_name):  # noqa: N802
    nodes = [  # LayerNorm subgraph
        helper.make_node("Shape", ["input_ids"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape1_out", "indices_0"], ["gather0_out"], "gather0"),
        (
            helper.make_node("Unsqueeze", ["gather0_out", "axes_0"], ["unsqueeze0_out"], "unsqueeze0")
            if opset_version == 13
            else helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0])
        ),
        helper.make_node("Shape", ["input_ids"], ["shape2_out"], "shape2"),
        helper.make_node("Gather", ["shape2_out", "indices_1"], ["gather1_out"], "gather1"),
        (
            helper.make_node("Unsqueeze", ["gather1_out", "axes_0"], ["unsqueeze1_out"], "unsqueeze1")
            if opset_version == 13
            else helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0])
        ),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out", "unsqueeze1_out"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node("Reshape", ["concat_out", "reshape_init"], ["reshape_out"], "reshape"),
        helper.make_node("Equal", ["reshape_out", "equal_init"], ["equal_out"], "equal"),
        helper.make_node("Where", ["equal_out", "where_init", "reshape_out"], ["where_out"], "where"),
        helper.make_node("Range", ["start_0", "gather1_out", "delta_1"], ["range_out"], "range"),
        (
            helper.make_node("Unsqueeze", ["range_out", "axes_0"], ["unsqueeze2_out"], "unsqueeze2")
            if opset_version == 13
            else helper.make_node("Unsqueeze", ["range_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0])
        ),
        helper.make_node("Expand", ["unsqueeze2_out", "where_out"], ["expand_out"], "expand"),
        helper.make_node("Gather", ["pos_embed", "expand_out"], ["pos_gather_out"], "pos_gather"),
        helper.make_node("Gather", ["word_embed", "input_ids"], ["word_gather_out"], "word_gather"),
        helper.make_node(
            "Add",
            ["word_gather_out", "pos_gather_out"],
            ["word_add_pos_out"],
            "word_add_pos",
        ),
        helper.make_node("Gather", ["seg_embed", "segment_ids"], ["seg_gather_out"], "seg_gather"),
        helper.make_node("Add", ["word_add_pos_out", "seg_gather_out"], ["add3_out"], "add3"),
        helper.make_node(
            "LayerNormalization",
            ["add3_out", "layer_norm_weight", "layer_norm_bias"],
            ["layernorm_out"],
            "layernorm",
            axis=-1,
            epsion=0.000009999999747378752,
        ),
        helper.make_node("Cast", ["input_mask"], ["mask_cast_out"], "mask_cast", to=6),
        (
            helper.make_node(
                "ReduceSum",
                ["mask_cast_out", "axes_1"],
                ["mask_index_out"],
                "mask_index",
                keepdims=0,
            )
            if opset_version == 13
            else helper.make_node(
                "ReduceSum",
                ["mask_cast_out"],
                ["mask_index_out"],
                "mask_index",
                axes=[1],
                keepdims=0,
            )
        ),
        helper.make_node(
            "Attention",
            ["layernorm_out", "qkv_weights", "qkv_bias", "mask_index_out"],
            ["att_out"],
            "att",
            domain="com.microsoft",
            num_heads=2,
        ),
        helper.make_node("MatMul", ["att_out", "matmul_weight"], ["matmul_out"], "matmul"),
        helper.make_node("Add", ["matmul_out", "add_bias"], ["add_out"], "add"),
        helper.make_node("Add", ["add_out", "layernorm_out"], ["add2_out"], "add2"),
    ]

    # hidden_size=4, num_heads=2, max_seq_length=3
    initializers = [  # initializers
        helper.make_tensor("indices_0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices_1", TensorProto.INT64, [], [1]),
        helper.make_tensor("start_0", TensorProto.INT64, [], [0]),
        helper.make_tensor("delta_1", TensorProto.INT64, [], [1]),
        helper.make_tensor(
            "word_embed",
            TensorProto.FLOAT,
            [2, 4],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        helper.make_tensor(
            "pos_embed",
            TensorProto.FLOAT,
            [4, 4],
            [
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        ),
        helper.make_tensor(
            "seg_embed",
            TensorProto.FLOAT,
            [2, 4],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        helper.make_tensor("layer_norm_weight", TensorProto.FLOAT, [4], [1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor("layer_norm_bias", TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),
        helper.make_tensor("qkv_weights", TensorProto.FLOAT, [4, 12], [0.1] * 4 * 12),
        helper.make_tensor("qkv_bias", TensorProto.FLOAT, [12], [0.1] * 12),
        helper.make_tensor(
            "matmul_weight",
            TensorProto.FLOAT,
            [4, 4],
            [
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        ),
        helper.make_tensor("add_bias", TensorProto.FLOAT, [4], [0.1, 0.2, 0.3, 0.4]),
        helper.make_tensor("reshape_init", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("equal_init", TensorProto.INT64, [2], [-1, -1]),
        helper.make_tensor("where_init", TensorProto.INT64, [2], [1, 1]),
        helper.make_tensor("axes_0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("axes_1", TensorProto.INT64, [1], [1]),
    ]

    graph = helper.make_graph(
        nodes,
        "EmbedLayerNorm_format6",  # name
        [  # inputs
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", 3]),
            helper.make_tensor_value_info("segment_ids", TensorProto.INT64, ["batch", 3]),
            helper.make_tensor_value_info("input_mask", TensorProto.INT64, ["batch", 3]),
        ],
        [  # outputs
            helper.make_tensor_value_info("add2_out", TensorProto.FLOAT, ["batch", 3, 4]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


def GenerateInitializers2(hidden_size):  # noqa: N802
    qkv_weights = [1.0] * hidden_size * (3 * hidden_size)

    initializers = [  # initializers
        helper.make_tensor(
            "word_embed",
            TensorProto.FLOAT,
            [2, hidden_size],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        helper.make_tensor(
            "pos_embed",
            TensorProto.FLOAT,
            [2, hidden_size],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        ),
        helper.make_tensor("indices_0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices_1", TensorProto.INT64, [], [1]),
        helper.make_tensor("start", TensorProto.INT64, [], [0]),
        helper.make_tensor("delta", TensorProto.INT64, [], [1]),
        helper.make_tensor("layer_norm_weight", TensorProto.FLOAT, [hidden_size], [1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor("layer_norm_bias", TensorProto.FLOAT, [hidden_size], [0.1, 0.2, 0.3, 0.4]),
        helper.make_tensor(
            "qkv_weights",
            TensorProto.FLOAT,
            [hidden_size, 3 * hidden_size],
            qkv_weights,
        ),
        helper.make_tensor(
            "qkv_bias",
            TensorProto.FLOAT,
            [3 * hidden_size],
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        ),
        helper.make_tensor(
            "matmul_weight",
            TensorProto.FLOAT,
            [hidden_size, hidden_size],
            [
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        ),
        helper.make_tensor("add_bias", TensorProto.FLOAT, [hidden_size], [0.1, 0.2, 0.3, 0.4]),
        helper.make_tensor("axes_0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("axes_1", TensorProto.INT64, [1], [1]),
    ]

    return initializers


def GenerateNodes2(attention_heads):  # noqa: N802
    nodes = [
        helper.make_node(
            "Gather",
            ["word_embed", "input_ids"],
            ["word_gather_out"],
            "word_gather",
            axis=0,
        ),
        helper.make_node("Shape", ["input_ids"], ["shape0_out"], "shape0"),
        helper.make_node("Gather", ["shape0_out", "indices_1"], ["gather0_out"], "gather0"),
        helper.make_node("Range", ["start", "gather0_out", "delta"], ["range0_out"], "range0"),
        (
            helper.make_node("Unsqueeze", ["range0_out", "axes_0"], ["unsqueeze0_out"], "unsqueeze0")
            if opset_version == 13
            else helper.make_node("Unsqueeze", ["range0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0])
        ),
        helper.make_node("Shape", ["input_ids"], ["shape1_out"], "shape1"),
        helper.make_node("Expand", ["unsqueeze0_out", "shape1_out"], ["expand_out"], "expand"),
        helper.make_node(
            "Gather",
            ["pos_embed", "expand_out"],
            ["pos_gather_out"],
            "pos_gather",
            axis=0,
        ),
        helper.make_node("Add", ["word_gather_out", "pos_gather_out"], ["add1_out"], "add1"),
        helper.make_node(
            "LayerNormalization",
            ["add1_out", "layer_norm_weight", "layer_norm_bias"],
            ["layernorm_out"],
            "layernorm",
            axis=-1,
            epsion=0.000009999999747378752,
        ),
        helper.make_node("Cast", ["input_mask"], ["mask_cast_out"], "mask_cast", to=6),
        (
            helper.make_node(
                "ReduceSum",
                ["mask_cast_out", "axes_1"],
                ["mask_index_out"],
                "mask_index",
                keepdims=0,
            )
            if opset_version == 13
            else helper.make_node(
                "ReduceSum",
                ["mask_cast_out"],
                ["mask_index_out"],
                "mask_index",
                axes=[1],
                keepdims=0,
            )
        ),
        helper.make_node(
            "Attention",
            ["layernorm_out", "qkv_weights", "qkv_bias", "mask_index_out"],
            ["att_out"],
            "att",
            domain="com.microsoft",
            num_heads=attention_heads,
        ),
        helper.make_node("MatMul", ["att_out", "matmul_weight"], ["matmul_out"], "matmul"),
        helper.make_node("Add", ["matmul_out", "add_bias"], ["add2_out"], "add2"),
        helper.make_node("Add", ["add2_out", "layernorm_out"], ["add3_out"], "add3"),
    ]

    return nodes


def GenerateModel7(model_name):  # noqa: N802
    batch_size = 2
    hidden_size = 4
    attention_heads = 2
    sequence_length = 3

    nodes = GenerateNodes2(attention_heads)

    initializers = GenerateInitializers2(hidden_size)

    graph = helper.make_graph(
        nodes,
        "EmbedLayerNorm_format7",  # name
        [  # inputs
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [batch_size, sequence_length]),
            helper.make_tensor_value_info("input_mask", TensorProto.INT64, [batch_size, sequence_length]),
        ],
        [  # outputs
            helper.make_tensor_value_info(
                "add3_out",
                TensorProto.FLOAT,
                [batch_size, sequence_length, hidden_size],
            ),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


def GenerateModel8(model_name):  # noqa: N802
    batch_size = -1
    hidden_size = 4
    attention_heads = 2
    sequence_length = -1

    nodes = GenerateNodes2(attention_heads)

    del nodes[5:7]
    del nodes[1:3]
    new_nodes = [
        helper.make_node("Shape", ["input_ids"], ["shape_out"], "shape"),
        helper.make_node("Gather", ["shape_out", "indices_1"], ["gather0_out"], "gather0"),
        helper.make_node("Expand", ["unsqueeze0_out", "shape_out"], ["expand_out"], "expand"),
    ]
    nodes = nodes + new_nodes

    initializers = GenerateInitializers2(hidden_size)

    graph = helper.make_graph(
        nodes,
        "EmbedLayerNorm_format8",  # name
        [  # inputs
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [batch_size, sequence_length]),
            helper.make_tensor_value_info("input_mask", TensorProto.INT64, [batch_size, sequence_length]),
        ],
        [  # outputs
            helper.make_tensor_value_info(
                "add3_out",
                TensorProto.FLOAT,
                [batch_size, sequence_length, hidden_size],
            ),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


def GenerateModel9(model_name):  # noqa: N802
    batch_size = -1
    hidden_size = 4
    attention_heads = 2
    sequence_length = -1

    nodes = GenerateNodes2(attention_heads)

    del nodes[10]
    del nodes[5:7]
    del nodes[1:3]
    new_nodes = [
        helper.make_node("Shape", ["input_ids"], ["shape_out"], "shape"),
        helper.make_node("Gather", ["shape_out", "indices_1"], ["gather0_out"], "gather0"),
        helper.make_node("Expand", ["unsqueeze0_out", "shape_out"], ["expand_out"], "expand"),
        helper.make_node("Gather", ["shape_out", "indices_0"], ["gather1_out"], "gather1"),
        helper.make_node("Gather", ["shape_out", "indices_1"], ["gather2_out"], "gather2"),
        (
            helper.make_node("Unsqueeze", ["gather1_out", "axes_0"], ["unsqueeze1_out"], "unsqueeze1")
            if opset_version == 13
            else helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0])
        ),
        (
            helper.make_node("Unsqueeze", ["gather2_out", "axes_0"], ["unsqueeze2_out"], "unsqueeze2")
            if opset_version == 13
            else helper.make_node("Unsqueeze", ["gather2_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0])
        ),
        helper.make_node(
            "Concat",
            ["unsqueeze1_out", "unsqueeze2_out"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node(
            "ConstantOfShape",
            ["concat_out"],
            ["constant_of_shape_out"],
            "constant_of_shape",
            value=helper.make_tensor("mask_shape", TensorProto.FLOAT, [1], [1.0]),
        ),
        helper.make_node("Cast", ["constant_of_shape_out"], ["mask_cast_out"], "mask_cast", to=6),
    ]
    nodes = nodes + new_nodes

    initializers = GenerateInitializers2(hidden_size)

    graph = helper.make_graph(
        nodes,
        "EmbedLayerNorm_format9",  # name
        [  # inputs
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [batch_size, sequence_length]),
        ],
        [  # outputs
            helper.make_tensor_value_info(
                "add3_out",
                TensorProto.FLOAT,
                [batch_size, sequence_length, hidden_size],
            ),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if opset_version == 11:
    GenerateModel3("embed_layer_norm_format3.onnx", True)
    GenerateModel3("embed_layer_norm_format3_no_cast.onnx", False)
    GenerateModel5("embed_layer_norm_format5.onnx")
    GenerateModel6("embed_layer_norm_format6.onnx")
    GenerateModel7("embed_layer_norm_format7.onnx")  # distilbert
    GenerateModel8("embed_layer_norm_format8.onnx")  # distilbert & shape nodes integration with input mask
    GenerateModel9("embed_layer_norm_format9.onnx")  # distilbert & shape nodes integration without input mask
    GenerateMultipleEmbedModel("embed_layer_norm_multiple.onnx")
else:
    GenerateModel3("embed_layer_norm_format3_opset13.onnx", True)
    GenerateModel3("embed_layer_norm_format3_no_cast_opset13.onnx", False)
    GenerateModel5("embed_layer_norm_format5_opset13.onnx")
    GenerateModel6("embed_layer_norm_format6_opset13.onnx")
    GenerateModel7("embed_layer_norm_format7_opset13.onnx")
    GenerateModel8("embed_layer_norm_format8_opset13.onnx")
    GenerateModel9("embed_layer_norm_format9_opset13.onnx")
    GenerateMultipleEmbedModel("embed_layer_norm_multiple_opset13.onnx")
