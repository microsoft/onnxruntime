"""Generates scan_in_local_function.onnx.

A model with a Scan op inside a local function.  The function body uses
symbolic dims; the callsite provides concrete dims.  This exercises ORT
issue #27887: after function inlining ORT must rename the value_info
entries inside the Scan body so CUDA EP can use the shape annotations to
pre-allocate correctly-sized carry-state buffers.

Topology
--------
Main graph (concrete shapes)
    init:    float[4]
    seq:     float[3, 4]
    result, out_seq = local.RunAccum(init, seq)

Function local.RunAccum(fn_init: float[S], fn_seq: float[N, S])
    -> (fn_final: float[S], fn_out: float[N, S])

    Scan body (symbolic dim S throughout):
        carry_in, x_in -> temp = Add(carry_in, x_in)   [value_info: temp: float[S]]
                       -> carry_out = Identity(temp)
                       -> x_out     = Identity(temp)

Expected output for init=[0,0,0,0], seq=[[1,1,1,1],[2,2,2,2],[3,3,3,3]]:
    result  = [6, 6, 6, 6]
    out_seq = [[1,1,1,1], [3,3,3,3], [6,6,6,6]]

Usage
-----
    python scan_in_local_function.py
"""

from onnx import AttributeProto, TensorProto, checker, helper, save

# ── Scan body subgraph ──────────────────────────────────────────────────────
# carry_in, x_in : float[S]  (symbolic S)
# temp           : float[S]  (annotated in value_info — this is what the fix renames)
# carry_out, x_out : float[S]
scan_body = helper.make_graph(
    nodes=[
        helper.make_node("Add", ["carry_in", "x_in"], ["temp"], name="body_add"),
        helper.make_node("Identity", ["temp"], ["carry_out"], name="body_id_carry"),
        helper.make_node("Identity", ["temp"], ["x_out"], name="body_id_x"),
    ],
    name="scan_body",
    inputs=[
        helper.make_tensor_value_info("carry_in", TensorProto.FLOAT, ["S"]),
        helper.make_tensor_value_info("x_in", TensorProto.FLOAT, ["S"]),
    ],
    outputs=[
        helper.make_tensor_value_info("carry_out", TensorProto.FLOAT, ["S"]),
        helper.make_tensor_value_info("x_out", TensorProto.FLOAT, ["S"]),
    ],
    value_info=[
        # Intermediate value annotation: after inlining, ORT must rename
        # "temp" to match the renamed node output (e.g. "temp__1").
        # CUDA EP memory planner uses this entry to pre-allocate the buffer.
        helper.make_tensor_value_info("temp", TensorProto.FLOAT, ["S"]),
    ],
)

# ── Scan node inside the function ───────────────────────────────────────────
scan_node = helper.make_node(
    "Scan",
    inputs=["fn_init", "fn_seq"],
    outputs=["fn_final", "fn_out"],
    name="scan",
    num_scan_inputs=1,
)
body_attr = AttributeProto()
body_attr.name = "body"
body_attr.type = AttributeProto.GRAPH
body_attr.g.CopyFrom(scan_body)
scan_node.attribute.append(body_attr)

# ── Local function ──────────────────────────────────────────────────────────
func = helper.make_function(
    domain="local",
    fname="RunAccum",
    inputs=["fn_init", "fn_seq"],
    outputs=["fn_final", "fn_out"],
    nodes=[scan_node],
    opset_imports=[helper.make_opsetid("", 18)],
)

# ── Main graph (concrete dims at the callsite) ──────────────────────────────
main_graph = helper.make_graph(
    nodes=[
        helper.make_node(
            "RunAccum",
            inputs=["init", "seq"],
            outputs=["result", "out_seq"],
            domain="local",
        ),
    ],
    name="main",
    inputs=[
        helper.make_tensor_value_info("init", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("seq", TensorProto.FLOAT, [3, 4]),
    ],
    outputs=[
        helper.make_tensor_value_info("result", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("out_seq", TensorProto.FLOAT, [3, 4]),
    ],
)

model = helper.make_model(
    main_graph,
    opset_imports=[
        helper.make_opsetid("", 18),
        helper.make_opsetid("local", 1),
    ],
    functions=[func],
    ir_version=8,
)

checker.check_model(model)
save(model, "scan_in_local_function.onnx")
print("Generated scan_in_local_function.onnx")
