import onnx
from onnx import helper
from onnx import TensorProto

def save_model(graph, file_name):
  model = helper.make_model(graph)
  onnx.checker.check_model(model)
  onnx.save(model, file_name)

graph = helper.make_graph(
    [ # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node("Gather", ["shape1_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices2"], ["gather2_out"], "gather2", axis=0),
        helper.make_node("Gather", ["shape2_out", "indices3"], ["gather3_out"], "gather3", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather2_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),
        helper.make_node("Unsqueeze", ["gather3_out"], ["unsqueeze3_out"], "unsqueeze3", axes=[0]),

        helper.make_node("Concat", ["unsqueeze0_out", "a1", "unsqueeze2_out", "unsqueeze3_out", "a4"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  #name
    [  # inputs
        helper.make_tensor_value_info('SubgraphRoot', TensorProto.FLOAT, ['unk_0', 256, 'unk_2', 'unk_3']),
    ],
    [  # outputs
        helper.make_tensor_value_info('Result', TensorProto.FLOAT, ['unk_1', 128, 'unk_2', 'unk_3', 'unk_4']),
        helper.make_tensor_value_info('gather3_out', TensorProto.INT64, []),
    ],
    [  # initializers
        helper.make_tensor('a1', TensorProto.INT64, [1], [128]),
        helper.make_tensor('a4', TensorProto.INT64, [1], [-1]),
        helper.make_tensor('indices0', TensorProto.INT64, [], [0]),
        helper.make_tensor('indices2', TensorProto.INT64, [], [2]),
        helper.make_tensor('indices3', TensorProto.INT64, [], [3]),
    ]
)

save_model(graph, 'reshape_fusion_internal_nodes_reused.onnx')


graph = helper.make_graph(
    [ # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape0_out"], "shape0"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices1"], ["gather1_out"], "gather1", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),

        helper.make_node("Concat", ["unsqueeze0_out", "unsqueeze1_out", "a"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  #name
    [  # inputs
        helper.make_tensor_value_info('SubgraphRoot', TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info('Result', TensorProto.FLOAT, [10, 20, 'unk']),
        helper.make_tensor_value_info('gather0_out', TensorProto.INT64, []),
    ],
    [  # initializers
        helper.make_tensor('a', TensorProto.INT64, [1], [-1]),
        helper.make_tensor('indices0', TensorProto.INT64, [], [0]),
        helper.make_tensor('indices1', TensorProto.INT64, [], [1]),
    ]
)

save_model(graph, 'reshape_fusion_internal_node_is_graph_output.onnx')

graph = helper.make_graph(
    [ # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node("Gather", ["shape2_out", "indices2"], ["gather2_out"], "gather2", axis=0),
        helper.make_node("Unsqueeze", ["gather2_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),

        helper.make_node("Concat", ["a", "unsqueeze2_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  #name
    [  # inputs
        helper.make_tensor_value_info('SubgraphRoot', TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info('Result', TensorProto.FLOAT, ['unk_0', 'unk_1', 'unk_2']),
    ],
    [  # initializers
        helper.make_tensor('a', TensorProto.INT64, [2], [1, 200]),
        helper.make_tensor('indices2', TensorProto.INT64, [], [1]),
    ]
)

save_model(graph, 'reshape_fusion_multiple_values_in_initializer_tensor_1.onnx')

graph = helper.make_graph(
    [ # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node("Gather", ["shape2_out", "indices2"], ["gather2_out"], "gather2", axis=0),
        helper.make_node("Unsqueeze", ["gather2_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),

        helper.make_node("Concat", ["a", "unsqueeze2_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  #name
    [  # inputs
        helper.make_tensor_value_info('SubgraphRoot', TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info('Result', TensorProto.FLOAT, ['unk_0', 'unk_1', 'unk_2']),
    ],
    [  # initializers
        helper.make_tensor('a', TensorProto.INT64, [2], [1, 200]),
        helper.make_tensor('indices2', TensorProto.INT64, [], [2]),
    ]
)

save_model(graph, 'reshape_fusion_multiple_values_in_initializer_tensor_2.onnx')

graph = helper.make_graph(
    [ # nodes
        helper.make_node("Shape", ["AnotherInput"], ["shape2_out"], "shape2"),
        helper.make_node("Gather", ["shape2_out", "indices2"], ["gather2_out"], "gather2", axis=0),
        helper.make_node("Unsqueeze", ["gather2_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),

        helper.make_node("Concat", ["a", "unsqueeze2_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  #name
    [  # inputs
        helper.make_tensor_value_info('SubgraphRoot', TensorProto.FLOAT, [10, 20, 30]),
        helper.make_tensor_value_info('AnotherInput', TensorProto.FLOAT, ['input_dim_0', 'input_dim_1', 'input_dim_2']),
    ],
    [  # outputs
        helper.make_tensor_value_info('Result', TensorProto.FLOAT, ['unk_0', 'unk_1', 'unk_2']),
    ],
    [  # initializers
        helper.make_tensor('a', TensorProto.INT64, [2], [1, 200]),
        helper.make_tensor('indices2', TensorProto.INT64, [], [2]),
    ]
)

save_model(graph, 'reshape_fusion_input_is_graph_input.onnx')

graph = helper.make_graph(
    [ # nodes
        helper.make_node("Concat", ["a"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  #name
    [  # inputs
        helper.make_tensor_value_info('SubgraphRoot', TensorProto.FLOAT, [2, 3, 4]),
        helper.make_tensor_value_info('a', TensorProto.INT64, [3]),
    ],
    [  # outputs
        helper.make_tensor_value_info('Result', TensorProto.FLOAT, ['unk_0', 'unk_1', 'unk_2']),
    ],
    [  # initializers
        helper.make_tensor('a', TensorProto.INT64, [3], [1, 1, 2*3*4]),
    ]
)

save_model(graph, 'reshape_fusion_overridable_initializer.onnx')

graph = helper.make_graph(
    [ # nodes
        helper.make_node("Shape", ["query"], ["shape0_out"], "shape0"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Concat", ["a", "unsqueeze0_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["doc_word_mask", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  #name
    [  # inputs
        helper.make_tensor_value_info('query', TensorProto.FLOAT, [1, 50]),
        helper.make_tensor_value_info('doc_word_mask', TensorProto.FLOAT, [1, 200, 50]),
    ],
    [  # outputs
        helper.make_tensor_value_info('Result', TensorProto.FLOAT, [10, 20, 'unk']),
    ],
    [  # initializers
        helper.make_tensor('a', TensorProto.INT64, [1], [-1]),
        helper.make_tensor('indices0', TensorProto.INT64, [], [1]),
    ]
)

save_model(graph, 'reshape_fusion_with_graph_inputs.onnx')

graph = helper.make_graph(
    [ # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape0_out"], "shape0"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices1"], ["gather1_out"], "gather1", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),
        
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node("Squeeze", ["shape2_out"], ["squeeze_out"], "squeeze"),
        helper.make_node("Div", ["squeeze_out", "div_init"], ["div_out"], "div"),
        helper.make_node("Unsqueeze", ["div_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),

        helper.make_node("Concat", ["unsqueeze0_out", "unsqueeze1_out", "unsqueeze2_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  #name
    [  # inputs
        helper.make_tensor_value_info('SubgraphRoot', TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info('Result', TensorProto.FLOAT, [10, 20, 'unk']),
    ],
    [  # initializers
        helper.make_tensor('indices0', TensorProto.INT64, [], [0]),
        helper.make_tensor('indices1', TensorProto.INT64, [], [1]),
        helper.make_tensor('div_init', TensorProto.INT64, [], [1]),
    ]
)

save_model(graph, 'reshape_fusion_concat_subgraph.onnx')

graph = helper.make_graph(
    [ # nodes
        helper.make_node("Shape", ["SubgraphRoot"], ["shape0_out"], "shape0"),
        helper.make_node("Shape", ["SubgraphRoot"], ["shape1_out"], "shape1"),
        helper.make_node("Gather", ["shape0_out", "indices0"], ["gather0_out"], "gather0", axis=0),
        helper.make_node("Gather", ["shape1_out", "indices1"], ["gather1_out"], "gather1", axis=0),
        helper.make_node("Unsqueeze", ["gather0_out"], ["unsqueeze0_out"], "unsqueeze0", axes=[0]),
        helper.make_node("Unsqueeze", ["gather1_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[0]),
        
        helper.make_node("Shape", ["unsqueeze0_out"], ["dummy_out"], "dummy"),
        
        helper.make_node("Shape", ["SubgraphRoot"], ["shape2_out"], "shape2"),
        helper.make_node("Squeeze", ["shape2_out"], ["squeeze_out"], "squeeze"),
        helper.make_node("Div", ["squeeze_out", "div_init"], ["div_out"], "div"),
        helper.make_node("Unsqueeze", ["div_out"], ["unsqueeze2_out"], "unsqueeze2", axes=[0]),

        helper.make_node("Concat", ["unsqueeze0_out", "unsqueeze1_out", "unsqueeze2_out"], ["concat_out"], "concat", axis=0),
        helper.make_node("Reshape", ["SubgraphRoot", "concat_out"], ["Result"], "reshape"),
    ],
    "Reshape_Fusion",  #name
    [  # inputs
        helper.make_tensor_value_info('SubgraphRoot', TensorProto.FLOAT, [10, 20, 30]),
    ],
    [  # outputs
        helper.make_tensor_value_info('Result', TensorProto.FLOAT, [10, 20, 'unk']),
        helper.make_tensor_value_info('div_out', TensorProto.INT64, []),
    ],
    [  # initializers
        helper.make_tensor('indices0', TensorProto.INT64, [], [0]),
        helper.make_tensor('indices1', TensorProto.INT64, [], [1]),
        helper.make_tensor('div_init', TensorProto.INT64, [], [1]),
    ]
)

save_model(graph, 'reshape_fusion_concat_subgraph_multiple_outputs.onnx')