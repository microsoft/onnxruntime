from onnx import TensorProto, helper, save_model

# Add node with a subgraph that has no inputs or outputs.
# Graph::BuildConnections should remove and the list of subgraphs in Graph::Resolve should be updated.
# Other details here don't matter. Copied from ort_github_issue_10305.py
if_body = helper.make_graph(
    [
        # need to use main_graph_initializer in a way that can't be constant folded
        helper.make_node("Constant", inputs=[], outputs=["zero"], name="Constant", value_int=0),
    ],
    "if_branch_body",
    [
        # no explicit inputs
    ],
    [
        helper.make_tensor_value_info("zero", TensorProto.BOOL, [1]),
    ],
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
            "If",
            [],
            [],
            "If1",
            then_branch=if_body,
            else_branch=if_body,
        ),
    ],
    "Main_graph",
    [
        helper.make_tensor_value_info("input:0", TensorProto.FLOAT, [2, 2, 2]),
        helper.make_tensor_value_info("state_var_in", TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor_value_info("transpose:0", TensorProto.FLOAT, [2, 2]),
    ],
)

model = helper.make_model(graph_proto)
# model to repro issue is invalid. can't run checker.
# onnx.checker.check_model(model, True)
save_model(model, "icm-31000000518483.onnx")
