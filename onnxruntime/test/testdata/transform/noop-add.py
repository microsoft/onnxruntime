import onnx
from onnx import helper
from onnx import TensorProto, OperatorSetIdProto

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = 'com.microsoft'

opsets.append(msdomain)
kwargs={}
kwargs['opset_imports'] = opsets

def GenerateModel(model_name):
    nodes = [  # subgraph
        # float
        helper.make_node("Identity", ["X1"], ["id_1"], "id_1"),
        helper.make_node("Add", ["float_1", "id_1"], ["add_1"], "add_1"),     
        helper.make_node("Identity", ["add_1"], ["Y1"], "id_2"),
        # float_16
        helper.make_node("Identity", ["X2"], ["id_3"], "id_3"),
        helper.make_node("Add", ["float16_1", "id_3"], ["add_2"], "add_2"),
        helper.make_node("Identity", ["add_2"], ["Y2"], "id_4"),      
        # int64 - flip the input 0 and 1
        helper.make_node("Identity", ["X3"], ["id_5"], "id_5"),
        helper.make_node("Add", ["id_5", "int64_1"], ["add_3"], "add_3"),
        helper.make_node("Identity", ["add_3"], ["Y3"], "id_6"),
        # int64
        helper.make_node("Identity", ["X4"], ["id_7"], "id_7"),
        helper.make_node("Add", ["id_7", "int64_2"], ["add_4"], "add_4"),
        helper.make_node("Identity", ["add_4"], ["Y4"], "id_8"),
    ]

    inputs = [  # inputs
            helper.make_tensor_value_info('X1', TensorProto.FLOAT, ['M', 'K']),
            helper.make_tensor_value_info('X2', TensorProto.FLOAT16, ['M', 'K']),
            helper.make_tensor_value_info('X3', TensorProto.INT64, ['M', 'K']),
            helper.make_tensor_value_info('X4', TensorProto.INT64, ['M', 'K']),
        ]

    initializers = [
            helper.make_tensor('float_1', TensorProto.FLOAT, [1], [0.0]),
            helper.make_tensor('float16_1', TensorProto.FLOAT16, [1], [0]),
            # int64 - set tensor size to 0
            helper.make_tensor('int64_1', TensorProto.INT64, (), [0]),
            # higher rank
            helper.make_tensor('int64_2', TensorProto.INT64, [1,1,1], [0]),
        ]

    graph = helper.make_graph(
        nodes,
        "NoopAdd",  #name
        inputs,
        [  # outputs
            helper.make_tensor_value_info('Y1', TensorProto.FLOAT, ['M', 'K']),
            helper.make_tensor_value_info('Y2', TensorProto.FLOAT16, ['M', 'K']),
            helper.make_tensor_value_info('Y3', TensorProto.INT64, ['M', 'K']),
            helper.make_tensor_value_info('Y4', TensorProto.INT64, ['M', 'K', 1]),
        ],
        initializers)

    model = helper.make_model(graph, **kwargs)
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModel('noop-add.onnx')