import onnx
from onnx import helper
from onnx import TensorProto

# Loop body graph and usage of main_graph_initializer on this level
# (Note: When running the model, we may need to disable graph optimization so that main_graph_initializer won't go away after constant folding)
body = helper.make_graph(
    [
        helper.make_node("Add", ["sub_graph_initializer", "main_graph_initializer"], ["initializer_sum"], "Add1"),
        helper.make_node("Add", ["initializer_sum", "loop_state_in"], ["loop_state_out"], "Add2"),
    ],
    "Loop_body",
    [
        helper.make_tensor_value_info("iteration_num", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("subgraph_keep_going_in", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info("loop_state_in", TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor_value_info("subgraph_keep_going_in", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info("loop_state_out", TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor("sub_graph_initializer", TensorProto.FLOAT, [1], [1.0]),
    ]
)

# Create the main graph
graph_proto = helper.make_graph(
    [
        helper.make_node("Loop", ["max_trip_count", "keep_going", "state_var_in"], ["state_var_out"], "Loop1", body=body)
    ],
    "Main_graph",
    [
        helper.make_tensor_value_info("state_var_in", TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor_value_info("state_var_out", TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor("max_trip_count", TensorProto.INT64, [1], [1]),
        helper.make_tensor("main_graph_initializer", TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor("keep_going", TensorProto.BOOL, [1], [True]),
    ]
)

model = helper.make_model(graph_proto)
onnx.save(model, "main_const_initializer_in_subgraph.onnx")
