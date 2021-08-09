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
    nodes = [
        helper.make_node("Add", ["A", "B"], ["X"], "addition"),
    ]

    inputs = [
            helper.make_tensor_value_info('A', TensorProto.FLOAT16, ['M', 'K']),
            helper.make_tensor_value_info('B', TensorProto.FLOAT16, ['M', 'K']),
        ]

    outputs = [
        helper.make_tensor_value_info('X', TensorProto.FLOAT16, ['M', 'K']),
        helper.make_tensor_value_info('X', TensorProto.FLOAT16, ['M', 'K']),
    ]

    graph = helper.make_graph(
        nodes,
        "GraphOutputDeduplication",
        inputs,
        outputs,
        [])

    model = helper.make_model(graph, **kwargs)
    print("saving model")
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModel('duplicate_graph_outputs.onnx')
