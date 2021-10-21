import onnx
from onnx import helper
from onnx import TensorProto

if_body = helper.make_graph(
    [
        # need to use main_graph_initializer in a way that can't be constant folded
        helper.make_node("Add", ["state_var_in", "main_graph_initializer"], ["add_out"], "If_add"),
        helper.make_node("Cast", ["add_out"], ["output"], to=TensorProto.BOOL),
    ],
    "if_branch_body",
    [
        # no explicit inputs
    ],
    [
        helper.make_tensor_value_info('output', TensorProto.BOOL, [1]), # how is this getting a type of float?
    ])

# Loop body graph with If node and usage of main_graph_initializer on this level
body = helper.make_graph(
    [
        # Add node that can be constant folded. Creates NodeArg when created but that implicit usage of an outer scope
        # value main_graph_initializer goes away after constant folding
        helper.make_node("Add", ["sub_graph_initializer", "main_graph_initializer"], ["initializer_sum"], "Add1"),
        helper.make_node("Add", ["initializer_sum", "loop_state_in"], ["loop_state_out"], "Add2"),
        # If node to create usage of main_graph_initializer another level down
        helper.make_node("If", ["subgraph_keep_going_in"], ["subgraph_keep_going_out"], "If1",
                         then_branch=if_body, else_branch=if_body),
    ],
    "Loop_body",
    [
        helper.make_tensor_value_info('iteration_num', TensorProto.INT64, [1]),
        helper.make_tensor_value_info('subgraph_keep_going_in', TensorProto.BOOL, [1]),
        helper.make_tensor_value_info('loop_state_in', TensorProto.FLOAT, [1])
    ],
    [
        helper.make_tensor_value_info('subgraph_keep_going_out', TensorProto.BOOL, [1]),
        helper.make_tensor_value_info('loop_state_out', TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor('sub_graph_initializer', TensorProto.FLOAT, [1], [1.])
    ]
)

# Create the main graph
graph_proto = helper.make_graph(
    [
        helper.make_node("Loop", ["max_trip_count", "keep_going", "state_var_in"],
                         ["state_var_out"], "Loop1", body=body)
    ],
    "Main_graph",
    [
        helper.make_tensor_value_info('state_var_in', TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor_value_info('state_var_out', TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor('max_trip_count', TensorProto.INT64, [1], [1]),
        helper.make_tensor('main_graph_initializer', TensorProto.FLOAT, [1], [1.]),
        helper.make_tensor('keep_going', TensorProto.BOOL, [1], [True]),
    ]
)

model = helper.make_model(graph_proto)
onnx.save(model, 'ort_github_issue_4031.onnx')