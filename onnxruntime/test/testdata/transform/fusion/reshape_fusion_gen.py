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

