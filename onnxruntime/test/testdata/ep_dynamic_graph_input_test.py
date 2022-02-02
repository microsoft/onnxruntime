import onnx
from onnx import helper
from onnx import TensorProto


# Since NNAPI EP does not support dynamic shape input and we now switch from the approach of immediately rejecting
# the whole graph in NNAPI EP if it has a dynamic input to checking the dynamic shape at individual operator support check level,
# We have a separated test here using a graph with dynamic input that becomes fixed after a Resize
# Please see BaseOpBuilder::HasSupportedInputs in <repo_root>/onnxruntime/core/providers/nnapi/nnapi_builtin/builders/op_support_checker.cc
def GenerateModel(model_name):
    nodes = [
        helper.make_node("Resize", ["X", "", "", "Resize_1_sizes"], [
                         "Resize_1_output"], "resize_1", mode="cubic"),
        helper.make_node(
            "Add", ["Resize_1_output", "Add_2_input"], ["Y"], "add"),
    ]

    initializers = [
        helper.make_tensor('Resize_1_sizes', TensorProto.INT64, [
                           4], [1, 1, 3, 3]),
        helper.make_tensor('Add_2_input',  TensorProto.FLOAT, [1, 1, 3, 3], [
                           1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    ]

    inputs = [
        helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, ["1", "1", "N", "N"]),  # used dim_param here
    ]

    outputs = [
        helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 3, 3]),
    ]

    graph = helper.make_graph(
        nodes,
        "EP_Dynamic_Graph_Input_Test",
        inputs,
        outputs,
        initializers
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel('ep_dynamic_graph_input_test.onnx')
