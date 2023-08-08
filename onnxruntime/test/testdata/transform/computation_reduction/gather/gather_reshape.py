import onnx
from onnx import OperatorSetIdProto, TensorProto, helper

hidden = 1024
head = 16


def _create_model_proto(
    input_shapes, output_shapes, axis_to_gather, slice_dims, slices_values, shape_dims, shape_values, model_name
):
    # inputs and outputs
    inputs = [
        helper.make_tensor_value_info("input1", TensorProto.FLOAT, input_shapes),
    ]

    outputs = [
        helper.make_tensor_value_info("final_output", TensorProto.FLOAT, output_shapes),
    ]

    # initializers

    initializers = [
        helper.make_tensor("shape", TensorProto.INT64, shape_dims, shape_values),
        helper.make_tensor("slices", TensorProto.INT64, slice_dims, slices_values),
    ]

    # nodes
    nodes = [
        helper.make_node("Reshape", ["input1", "shape"], ["reshape_out"], "reshape1"),
        helper.make_node("Gather", ["reshape_out", "slices"], ["gather_out"], "gather", axis=axis_to_gather),
        helper.make_node("Identity", ["gather_out"], ["final_output"], "identity1"),
    ]

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        nodes,
        "test-model",
        inputs,
        outputs,
        initializers,
        "doc string",
    )

    opsets = []
    onnxdomain = OperatorSetIdProto()
    onnxdomain.version = 14
    onnxdomain.domain = ""  # The empty string ("") or absence of this field implies the operator
    # set that is defined as part of the ONNX specification.
    opsets.append(onnxdomain)

    msdomain = OperatorSetIdProto()
    msdomain.version = 1
    msdomain.domain = "com.microsoft"

    opsets.append(msdomain)
    kwargs = {}
    kwargs["opset_imports"] = opsets

    model_def = helper.make_model(graph_def, producer_name="onnx-example", **kwargs)
    final_model = onnx.shape_inference.infer_shapes(model_def)
    onnx.save(final_model, model_name + ".onnx")


input_shapes1 = ["batch_size", "sequence_length", hidden]
_create_model_proto(
    input_shapes1, ["sequence_length", 16, 64], 0, [], [1], [4], [0, 0, 16, 64], "gather_reshape_scalar_batch_dim"
)
_create_model_proto(
    input_shapes1, [1, "sequence_length", 16, 64], 0, [1], [1], [4], [0, 0, 16, 64], "gather_reshape_batch_dim"
)
_create_model_proto(
    input_shapes1, ["batch_size", 16, 64], 1, [], [1], [4], [0, 0, 16, 64], "gather_reshape_scalar_seqlen_dim"
)
_create_model_proto(
    input_shapes1, ["batch_size", 1, 16, 64], 1, [1], [1], [4], [0, 0, 16, 64], "gather_reshape_seqlen_dim"
)


input_shapes2 = ["batch_size", 128, hidden]
_create_model_proto(
    input_shapes2,
    ["batch_size", 31, 16, 64],
    1,
    [31],
    [i for i in range(31)],
    [4],
    [0, 128, 16, 64],
    "gather_reshape_seqlen_dim2",
)
