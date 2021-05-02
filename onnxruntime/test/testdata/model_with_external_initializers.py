import onnx
import numpy as np
from onnx import helper
from onnx import TensorProto
from onnx.numpy_helper import from_array
from onnx.external_data_helper import set_external_data


def create_external_data_tensor(value, tensor_name):  # type: (List[Any], Text) -> TensorProto
    tensor = from_array(np.array(value))
    tensor.name = tensor_name
    tensor_filename = "{}.bin".format(tensor_name)
    set_external_data(tensor, location=tensor_filename)

    with open(os.path.join(tensor_filename), 'wb') as data_file:
        data_file.write(tensor.raw_data)
    tensor.ClearField('raw_data')
    tensor.data_location = onnx.TensorProto.EXTERNAL
    return tensor


def GenerateModel(model_name):
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])

    # Create second input (ValueInfoProto)
    Pads = helper.make_tensor_value_info('Pads', TensorProto.INT64, [4])

    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        'Pad', # node name
        ['X', 'Pads'], # inputs
        ['Y'], # outputs
        mode='constant', # Attributes
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model",
        [X, Pads],
        [Y],
        [create_external_data_tensor([0, 0, 1, 1, ], "Pads")]
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def,
                                  producer_name='onnx-example')

    print('The ir_version in model: {}\n'.format(model_def.ir_version))
    print('The producer_name in model: {}\n'.format(model_def.producer_name))
    print('The graph in model:\n{}'.format(model_def.graph))
    onnx.checker.check_model(model_def)
    print('The model is checked!')
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


if __name__ == "__main__":
    GenerateModel('model_with_external_initializers.onnx')
