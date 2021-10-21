import onnx
from onnx import helper
from onnx import TensorProto


# Create a model with metadata to test ORT conversion
def GenerateModel(model_name):
    nodes = [
        helper.make_node("Sigmoid", ["X"], ["Y"], "sigmoid"),
    ]

    graph = helper.make_graph(
        nodes,
        "NNAPI_Internal_uint8_Test",
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
    )

    model = helper.make_model(graph)

    # Add meta data
    model.doc_string = 'This is doc_string'
    model.producer_name = 'TensorTorch'
    model.model_version = 12345
    model.domain = 'ai.onnx.ml'
    helper.set_model_props(
        model,
        {
            'I am key 1!': 'I am value 1!',
            '': 'Value for empty key!',
            'Key for empty value!': '',
        }
    )
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel('model_with_metadata.onnx')
