import onnx
from onnx import helper
from onnx import TensorProto, OperatorSetIdProto
from enum import Enum

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 13
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = 'com.microsoft'

opsets.append(msdomain)
kwargs={}
kwargs['opset_imports'] = opsets

def GenerateModelCast(model_name):
    nodes = [  # subgraph
        # float
        helper.make_node("Identity", ["X"], ["x"], "identity_0"),
        helper.make_node("Cast", ["x"], ["casted_x"], "cast_0", to = TensorProto.FLOAT16),
        helper.make_node("Shape", ["casted_x"], ["shape_x0"], "shape_0"),
        helper.make_node("Cast", ["x"], ["casted_x_1"], "cast_1", to = TensorProto.FLOAT),
        helper.make_node("Shape", ["casted_x_1"], ["shape_x1"], "shape_1"),
        helper.make_node("Identity", ["casted_x_1"], ["casted_x_2"], "identity_1"),
        helper.make_node("Cast", ["casted_x_2"], ["casted_x_3"], "cast_2", to = TensorProto.FLOAT16),
        helper.make_node("Shape", ["casted_x_3"], ["shape_x2"], "shape_2"),
    ]

    inputs = [  # inputs
            helper.make_tensor_value_info('X', TensorProto.BOOL, ['M', 'K']),
        ]

    initializers = []

    graph = helper.make_graph(
        nodes,
        "NotWhere",  #name
        inputs,
        [  # outputs
            helper.make_tensor_value_info('shape_x0', TensorProto.INT64, [2]),
            helper.make_tensor_value_info('shape_x1', TensorProto.INT64, [2]),
            helper.make_tensor_value_info('casted_x_3', TensorProto.FLOAT16, ['M', 'K']),
            helper.make_tensor_value_info('shape_x2', TensorProto.INT64, [2]),
        ],
        initializers)

    model = helper.make_model(graph, **kwargs)
    onnx.save(model, model_name)

def GenerateModelTranspose(model_name):
    nodes = [  # subgraph
        # float
        helper.make_node("Identity", ["X"], ["x"], "identity_0"),
        helper.make_node("Transpose", ["x"], ["trx"], "transpose0"),
        helper.make_node("Shape", ["trx"], ["shape_x0"], "shape_0"),
        helper.make_node("Transpose", ["x"], ["trx_1"], "transpose1", ),
        helper.make_node("Shape", ["trx_1"], ["shape_x1"], "shape_1"),
        helper.make_node("Identity", ["trx_1"], ["trx_2"], "identity_1"),
        helper.make_node("Transpose", ["trx_2"], ["trx_3"], "transpose2"),
        helper.make_node("Shape", ["trx_3"], ["shape_x2"], "shape_2"),
        helper.make_node("Transpose", ["x"], ["trx_4"], "transpose3", perm=[1,0]),
        helper.make_node("Shape", ["trx_4"], ["shape_x3"], "shape_3"),
    ]

    inputs = [  # inputs
            helper.make_tensor_value_info('X', TensorProto.FLOAT16, ['M', 'K']),
        ]

    initializers = []

    graph = helper.make_graph(
        nodes,
        "NotWhere",  #name
        inputs,
        [  # outputs
            helper.make_tensor_value_info('shape_x0', TensorProto.INT64, [2]),
            helper.make_tensor_value_info('shape_x1', TensorProto.INT64, [2]),
            helper.make_tensor_value_info('trx_3', TensorProto.FLOAT16, ['M', 'K']),
            helper.make_tensor_value_info('shape_x2', TensorProto.INT64, [2]),
            helper.make_tensor_value_info('shape_x3', TensorProto.INT64, [2]),
        ],
        initializers)

    model = helper.make_model(graph, **kwargs)
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModelCast('shape_opt_cast.onnx')
    GenerateModelTranspose('shape_opt_transpose.onnx')
