import onnx
from onnx import TensorProto, helper

# Loop is so the Tranpose output is used in a subgraph
loop_body = helper.make_graph(
    [
        helper.make_node("Add", ["transpose:0", "loop_state_in"], ["loop_state_out"], "Add1"),
    ],
    "Loop_body",
    [
        helper.make_tensor_value_info("iteration_num", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("subgraph_keep_going_in", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info("loop_state_in", TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor_value_info("subgraph_keep_going_in", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info("loop_state_out", TensorProto.FLOAT, [2, 2, 2]),
    ],
    [],
)

# Create the main graph
graph_proto = helper.make_graph(
    [
        # add a Transpose that can be moved past the Slice
        helper.make_node(
            "Transpose",
            inputs=["input:0"],
            outputs=["transpose:0"],
            name="transpose0",
            perm=[1, 0, 2],
        ),
        helper.make_node(
            "Slice",
            inputs=["transpose:0", "start", "end"],
            outputs=["strided_slice:0"],
            name="slice0",
        ),
        helper.make_node(
            "Squeeze",
            inputs=["strided_slice:0", "start"],
            outputs=["out:0"],
            name="squeeze0",
        ),
        helper.make_node(
            "Loop",
            ["max_trip_count", "subgraph_keep_going_in", "state_var_in"],
            ["out:1"],
            "Loop1",
            body=loop_body,
        ),
    ],
    "Main_graph",
    [
        helper.make_tensor_value_info("input:0", TensorProto.FLOAT, [2, 2, 2]),
        helper.make_tensor_value_info("state_var_in", TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor_value_info("out:0", TensorProto.FLOAT, [2, 2]),
        helper.make_tensor_value_info("out:1", TensorProto.FLOAT, [2, 2, 2]),
    ],
    [
        helper.make_tensor("start", TensorProto.INT64, [1], [0]),
        helper.make_tensor("end", TensorProto.INT64, [1], [1]),
        helper.make_tensor("max_trip_count", TensorProto.INT64, [1], [1]),
        helper.make_tensor("subgraph_keep_going_in", TensorProto.BOOL, [1], [1]),
    ],
)

model = helper.make_model(graph_proto)
onnx.checker.check_model(model, True)
onnx.save(model, "ort_github_issue_10305.onnx")
