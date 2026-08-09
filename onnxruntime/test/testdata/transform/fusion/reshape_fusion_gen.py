import onnx
from onnx import TensorProto, helper


def save_model(graph, file_name):
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    onnx.save(model, file_name)


graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node("Gather", ["shape1_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices2"], ["gather2_out"], "gather2", axis=0),
        helper.make_node("Gather", ["shape2_out", "indices3"], ["gather3_out"], "gather3", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather2_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),
        helper.make_node("Unsqueeze", ["gather3_out"], ["unsqueeze3_out"], "unsqueeze3", axes=[0]),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out", "a1", "unsqueeze2_out", "unsqueeze3_out", "a4"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, ["unk_0", 256, "unk_2", "unk_3"]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, ["unk_1", 128, "unk_2", "unk_3", "unk_4"]),
        helper.make_tensor_value_info("gather3_out", TensorProto.INT64, []),
    ],
    [  # initializers
        helper.make_tensor("a1", TensorProto.INT64, [1], [128]),
        helper.make_tensor("a4", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("indices0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices2", TensorProto.INT64, [], [2]),
        helper.make_tensor("indices3", TensorProto.INT64, [], [3]),
    ],
)

save_model(graph, "reshape_fusion_internal_nodes_reused.onnx")


graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape0_out"], "shape0"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices1"], ["gather1_out"], "gather1", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out", "unsqueeze1_out", "a"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, [10, 20, "unk"]),
        helper.make_tensor_value_info("gather0_out", TensorProto.INT64, []),
    ],
    [  # initializers
        helper.make_tensor("a", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("indices0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices1", TensorProto.INT64, [], [1]),
    ],
)

save_model(graph, "reshape_fusion_internal_node_is_graph_output.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node("Gather", ["shape2_out", "indices2"], ["gather2_out"], "gather2", axis=0),
        helper.make_node("Unsqueeze", ["gather2_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),
        helper.make_node("Concat", ["a", "unsqueeze2_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, ["unk_0", "unk_1", "unk_2"]),
    ],
    [  # initializers
        helper.make_tensor("a", TensorProto.INT64, [2], [1, 200]),
        helper.make_tensor("indices2", TensorProto.INT64, [], [1]),
    ],
)

save_model(graph, "reshape_fusion_multiple_values_in_initializer_tensor_1.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node("Gather", ["shape2_out", "indices2"], ["gather2_out"], "gather2", axis=0),
        helper.make_node("Unsqueeze", ["gather2_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),
        helper.make_node("Concat", ["a", "unsqueeze2_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, ["unk_0", "unk_1", "unk_2"]),
    ],
    [  # initializers
        helper.make_tensor("a", TensorProto.INT64, [2], [1, 200]),
        helper.make_tensor("indices2", TensorProto.INT64, [], [2]),
    ],
)

save_model(graph, "reshape_fusion_multiple_values_in_initializer_tensor_2.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["AnotherInput"], ["shape2_out"], "shape2"),
        helper.make_node("Gather", ["shape2_out", "indices2"], ["gather2_out"], "gather2", axis=0),
        helper.make_node("Unsqueeze", ["gather2_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),
        helper.make_node("Concat", ["a", "unsqueeze2_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [10, 20, 30]),
        helper.make_tensor_value_info(
            "AnotherInput",
            TensorProto.FLOAT,
            ["input_dim_0", "input_dim_1", "input_dim_2"],
        ),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, ["unk_0", "unk_1", "unk_2"]),
    ],
    [  # initializers
        helper.make_tensor("a", TensorProto.INT64, [2], [1, 200]),
        helper.make_tensor("indices2", TensorProto.INT64, [], [2]),
    ],
)

save_model(graph, "reshape_fusion_input_is_graph_input.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Concat", ["a"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [2, 3, 4]),
        helper.make_tensor_value_info("a", TensorProto.INT64, [3]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, ["unk_0", "unk_1", "unk_2"]),
    ],
    [  # initializers
        helper.make_tensor("a", TensorProto.INT64, [3], [1, 1, 2 * 3 * 4]),
    ],
)

save_model(graph, "reshape_fusion_overridable_initializer.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["query"], ["shape0_out"], "shape0"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Concat", ["a", "unsqueeze0_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["doc_word_mask", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [1, 50]),
        helper.make_tensor_value_info("doc_word_mask", TensorProto.FLOAT, [1, 200, 50]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, [10, 20, "unk"]),
    ],
    [  # initializers
        helper.make_tensor("a", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("indices0", TensorProto.INT64, [], [1]),
    ],
)

save_model(graph, "reshape_fusion_with_graph_inputs.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape0_out"], "shape0"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices1"], ["gather1_out"], "gather1", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node(
            "Slice",
            ["shape2_out", "slice_starts", "slice_ends"],
            ["slice_out"],
            "slice1",
        ),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out", "unsqueeze1_out", "slice_out"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, [10, 20, "unk"]),
    ],
    [  # initializers
        helper.make_tensor("indices0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices1", TensorProto.INT64, [], [1]),
        helper.make_tensor("slice_starts", TensorProto.INT64, [1], [2]),
        helper.make_tensor("slice_ends", TensorProto.INT64, [1], [3]),
    ],
)

save_model(graph, "reshape_fusion_concat_subgraph.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape0_out"], "shape0"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices1"], ["gather1_out"], "gather1", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node("Slice", ["shape2_out"], ["slice_out"], "slice1", starts=[2], ends=[3]),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out", "unsqueeze1_out", "slice_out"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, [10, 20, "unk"]),
    ],
    [  # initializers
        helper.make_tensor("indices0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices1", TensorProto.INT64, [], [1]),
    ],
)
# Save this model without checking
onnx.save(helper.make_model(graph), "reshape_fusion_with_slice1.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape0_out"], "shape0"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices1"], ["gather1_out"], "gather1", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),
        helper.make_node("Shape", ["unsqueeze0_out"], ["dummy_out"], "dummy"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node(
            "Slice",
            ["shape2_out", "slice_starts", "slice_ends"],
            ["slice_out"],
            "slice1",
        ),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out", "unsqueeze1_out", "slice_out"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, [10, 20, "unk"]),
        helper.make_tensor_value_info("slice_out", TensorProto.INT64, [1]),
    ],
    [  # initializers
        helper.make_tensor("indices0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices1", TensorProto.INT64, [], [1]),
        helper.make_tensor("slice_starts", TensorProto.INT64, [1], [2]),
        helper.make_tensor("slice_ends", TensorProto.INT64, [1], [3]),
    ],
)

save_model(graph, "reshape_fusion_concat_subgraph_multiple_outputs.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape0_out"], "shape0"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices1"], ["gather1_out"], "gather1", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node(
            "Slice",
            ["shape2_out", "slice_starts", "slice_ends"],
            ["slice_out"],
            "slice1",
        ),
        helper.make_node("Pad", ["slice_out", "pads"], ["pad0_out"], "pad0", mode="constant"),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out", "unsqueeze1_out", "pad0_out"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [10, 20, 30]),
    ],
    [helper.make_tensor_value_info("Result", TensorProto.FLOAT, [10, 20, "unk"])],  # outputs
    [  # initializers
        helper.make_tensor("indices0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices1", TensorProto.INT64, [], [1]),
        helper.make_tensor("pads", TensorProto.INT64, [2], [1, 0]),
        helper.make_tensor("slice_starts", TensorProto.INT64, [1], [2]),
        helper.make_tensor("slice_ends", TensorProto.INT64, [1], [3]),
    ],
)

save_model(graph, "reshape_fusion_concat_subgraph_not_triggered.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape0_out"], "shape0"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices1"], ["gather1_out"], "gather1", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node(
            "Slice",
            ["shape2_out", "slice_starts", "slice_ends"],
            ["slice_out"],
            "slice1",
        ),
        helper.make_node("Squeeze", ["slice_out"], ["squeeze0_out"], "squeeze0", axes=[0]),
        helper.make_node("Div", ["squeeze0_out", "div_init"], ["div_out"], "div"),
        helper.make_node("Unsqueeze", ["div_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out", "unsqueeze1_out", "unsqueeze2_out"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [10, 20, 30]),
    ],
    [helper.make_tensor_value_info("Result", TensorProto.FLOAT, [10, 20, "unk"])],  # outputs
    [  # initializers
        helper.make_tensor("indices0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices1", TensorProto.INT64, [], [1]),
        helper.make_tensor("div_init", TensorProto.INT64, [], [1]),
        helper.make_tensor("slice_starts", TensorProto.INT64, [1], [2]),
        helper.make_tensor("slice_ends", TensorProto.INT64, [1], [3]),
    ],
)

save_model(graph, "reshape_fusion_concat_subgraph_div.onnx")

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape0_out"], "shape0"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices1"], ["gather1_out"], "gather1", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node(
            "Slice",
            ["shape2_out", "slice_starts_0", "slice_ends_0"],
            ["slice0_out"],
            "slice0",
        ),
        helper.make_node("Squeeze", ["slice0_out"], ["squeeze0_out"], "squeeze0", axes=[0]),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape3_out"], "shape3"),
        helper.make_node(
            "Slice",
            ["shape3_out", "slice_starts_1", "slice_ends_1"],
            ["slice1_out"],
            "slice1",
        ),
        helper.make_node("Squeeze", ["slice1_out"], ["squeeze1_out"], "squeeze1", axes=[0]),
        helper.make_node("Mul", ["squeeze0_out", "squeeze1_out"], ["mul_out"], "mul"),
        helper.make_node("Unsqueeze", ["mul_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out", "unsqueeze1_out", "unsqueeze2_out"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("SubgraphRoot", TensorProto.FLOAT, [10, 20, 30]),
    ],
    [helper.make_tensor_value_info("Result", TensorProto.FLOAT, [10, 20, "unk"])],  # outputs
    [  # initializers
        helper.make_tensor("indices0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices1", TensorProto.INT64, [], [1]),
        helper.make_tensor("slice_starts_0", TensorProto.INT64, [1], [2]),
        helper.make_tensor("slice_ends_0", TensorProto.INT64, [1], [3]),
        helper.make_tensor("slice_starts_1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("slice_ends_1", TensorProto.INT64, [1], [2]),
    ],
)

save_model(graph, "reshape_fusion_concat_subgraph_mul.onnx")

matmul_weights = [
    -0.04888916015625,
    0.0143280029296875,
    0.066650390625,
    -0.0343017578125,
    -0.0010356903076171875,
    -0.00048232078552246094,
    0.07470703125,
    -0.04736328125,
    0.01454925537109375,
    -0.0086669921875,
    -0.051971435546875,
    -0.0201568603515625,
    0.040435791015625,
    -0.019256591796875,
    0.0205078125,
    0.0111541748046875,
    0.0071868896484375,
    -0.0298309326171875,
    -0.0306549072265625,
    -0.0225372314453125,
    -0.04193115234375,
    0.07073974609375,
    -0.048065185546875,
    0.0198822021484375,
    -0.035552978515625,
    -0.022796630859375,
    0.03839111328125,
    0.007099151611328125,
    -0.0080108642578125,
    -0.0017957687377929688,
    0.0266265869140625,
    -0.028289794921875,
    0.0032901763916015625,
    0.0208740234375,
    -0.01529693603515625,
    -0.046600341796875,
    -0.034637451171875,
    0.011322021484375,
    -0.026458740234375,
    0.04656982421875,
    -0.0091705322265625,
    0.017913818359375,
    -0.019256591796875,
    -0.001216888427734375,
    -0.08245849609375,
    -0.023162841796875,
    -0.04132080078125,
    -0.03363037109375,
    0.0029315948486328125,
    0.03173828125,
    -0.004024505615234375,
    0.04534912109375,
    -0.0036163330078125,
    -0.03912353515625,
    -0.00800323486328125,
    0.058197021484375,
    0.05572509765625,
    0.01165771484375,
    0.06756591796875,
    0.05816650390625,
    -0.0654296875,
    -0.0241851806640625,
    0.0205535888671875,
    -0.031707763671875,
]

add_weight = [
    -0.23681640625,
    -0.16552734375,
    0.2191162109375,
    -0.1756591796875,
    -0.03460693359375,
    -0.05316162109375,
    -0.336181640625,
    -0.253662109375,
]

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Add", ["Input", "Bias"], ["add0_out"], "add0"),
        helper.make_node("Shape", ["add0_out"], ["shape0_out"], "shape0"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node(
            "Concat",
            ["unsqueeze0_out", "dim_-1", "dim_2", "dim_4"],
            ["concat_out"],
            "concat",
            axis=0,
        ),
        helper.make_node("MatMul", ["add0_out", "matmul_weight"], ["matmul_out"], "matmul"),
        helper.make_node("Add", ["matmul_out", "add_weight"], ["add1_out"], "add1"),
        helper.make_node("Reshape", ["add1_out", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  # name
    [  # inputs
        helper.make_tensor_value_info("Input", TensorProto.FLOAT, [1, 8]),
    ],
    [  # outputs
        helper.make_tensor_value_info("Result", TensorProto.FLOAT, [1, -1, 2, 4]),
    ],
    [  # initializers
        helper.make_tensor("Bias", TensorProto.FLOAT, [8], add_weight),
        helper.make_tensor("dim_-1", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("dim_2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("dim_4", TensorProto.INT64, [1], [4]),
        helper.make_tensor("indices0", TensorProto.INT64, [], [0]),
        helper.make_tensor("matmul_weight", TensorProto.FLOAT, [8, 8], matmul_weights),
        helper.make_tensor("add_weight", TensorProto.FLOAT, [8], add_weight),
    ],
)

save_model(graph, "reshape_fusion_distillbert.onnx")
