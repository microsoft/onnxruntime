import onnx
from onnx import TensorProto, helper

# CoreML EP currently handles a special case for supporting ArgMax followed by a Cast to int32.
# Please see <repo_root>/onnxruntime/core/providers/coreml/builders/impl/argmax_op_builder.cc and
# <repo_root>/onnxruntime/core/providers/coreml/builders/impl/cast_op_builder.cc.
# This script generates graphs for these cases:
# - An ArgMax followed by a supported Cast to int32 type
# - An ArgMax followed by an unsupported Cast to a type other than int32


def GenerateModel(model_name, cast_to_dtype):  # noqa: N802
    nodes = [
        helper.make_node("ArgMax", ["X"], ["argmax_output_int64"], "argmax", axis=1, keepdims=1),
        helper.make_node("Cast", ["argmax_output_int64"], ["Y"], "cast", to=cast_to_dtype),
    ]

    graph = helper.make_graph(
        nodes,
        "CoreML_ArgMax_Cast_Test",
        [  # input
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2, 2]),
        ],
        [  # output
            helper.make_tensor_value_info("Y", cast_to_dtype, [3, 1, 2]),
        ],
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel("coreml_argmax_cast_test.onnx", TensorProto.INT32)
    GenerateModel("coreml_argmax_unsupported_cast_test.onnx", TensorProto.UINT32)
