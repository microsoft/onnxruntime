import onnx
from onnx import OperatorSetIdProto, TensorProto, helper


def _create_model_proto(output_shapes, axis_to_gather, slice_dims, slices_values, model_name):
    # inputs and outputs
    hidden = 1024
    inputs = [
        helper.make_tensor_value_info("input1", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden]),
        helper.make_tensor_value_info("input2", TensorProto.FLOAT, ["batch_size", hidden, "sequence_length"]),
    ]

    outputs = [
        helper.make_tensor_value_info("final_output", TensorProto.FLOAT, output_shapes),
    ]

    # initializers

    initializers = [
        helper.make_tensor("slices", TensorProto.INT64, slice_dims, slices_values),
    ]

    # nodes
    nodes = [
        helper.make_node("MatMul", ["input1", "input2"], ["m1_out"], "m1"),
        helper.make_node("Gather", ["m1_out", "slices"], ["gather_out"], "gather", axis=axis_to_gather),
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


_create_model_proto(["sequence_length", "sequence_length"], 0, [], [1], "gather_matmul_scalar_batch_dim")
_create_model_proto([1, "sequence_length", "sequence_length"], 0, [1], [1], "gather_matmul_batch_dim")
_create_model_proto(["batch_size", "sequence_length"], 1, [], [1], "gather_matmul_scalar_second_last_dim")
_create_model_proto(["batch_size", 1, "sequence_length"], 1, [1], [1], "gather_matmul_second_last_dim")
_create_model_proto(["batch_size", "sequence_length"], 2, [], [1], "gather_matmul_scalar_last_dim")
_create_model_proto(["batch_size", "sequence_length", 1], 2, [1], [1], "gather_matmul_last_dim")
