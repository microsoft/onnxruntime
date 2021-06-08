import onnx
from onnx import helper
from onnx import TensorProto

# CoreML EP currently handles a special case for supporting ArgMax op
# Please see in <repo_root>/onnxruntime/core/providers/coreml/builders/impl/argmax_op_builder.cc and
# <repo_root>/onnxruntime/core/providers/coreml/builders/impl/cast_op_builder.cc
# We have a separated test for this special case: An ArgMax followed by a Cast to int32 type


def GenerateModel(model_name):
    nodes = [
        helper.make_node("ArgMax", ["X"], [
                         "argmax_output_int64"], "argmax", axis=1, keepdims=1),
        helper.make_node("Cast", ["argmax_output_int64"], [
                         "Y"], "cast", to=6),  # cast to int32 type
    ]

    input = [
        helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2, 2]),
    ]

    output = [
        helper.make_tensor_value_info('Y', TensorProto.INT32, [3, 1, 2]),
    ]

    graph = helper.make_graph(
        nodes,
        "CoreML_ArgMax_Cast_Test",
        input,
        output
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel('coreml_argmax_cast_test.onnx')
