import os
import sys
from ort_flatbuffers.onnxruntime.experimental.fbs import InferenceSession
# from ort_flatbuffers.onnxruntime.experimental.fbs import Model
from ort_flatbuffers.onnxruntime.experimental.fbs import Graph
from ort_flatbuffers.onnxruntime.experimental.fbs import AttributeType
# from ort_flatbuffers.onnxruntime.experimental.fbs import Node

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ort_flatbuffers'))


def add_required_op(required_ops, op_type: str, raw_domain: str, opset: int):
    script_path = os.path.dirname(os.path.realpath(__file__))
    ci_build_py_path = os.path.abspath(os.path.join(script_path, '..', 'ci_build'))
    sys.path.append(ci_build_py_path)
    import op_registration_utils  # tools/ci_build/op_registration_utils.py

    domain = op_registration_utils.map_domain(raw_domain)
    operators = set([op_type])

    if domain not in required_ops:
        required_ops[domain] = {opset: operators}
    elif opset not in required_ops[domain]:
        required_ops[domain][opset] = operators
    else:
        required_ops[domain][opset].update(operators)


def read_nodes_from_graph(graph: Graph, required_ops):
    # print(dir(graph.Nodes(0)))
    for i in range(0, graph.NodesLength()):
        node = graph.Nodes(i)
        add_required_op(required_ops, node.OpType().decode(), node.Domain().decode(), node.SinceVersion())

        # Read all the attributes
        for j in range(0, node.AttributesLength()):
            attr = node.Attributes(j)
            attr_type = attr.Type()
            if attr_type == AttributeType.AttributeType.GRAPH:
                read_nodes_from_graph(attr.G(), required_ops)
            elif attr_type == AttributeType.AttributeType.GRAPHS:
                for k in range(0, attr.GraphsLength()):
                    read_nodes_from_graph(attr.Graphs(k), required_ops)



if __name__ == "__main__":
    # file = open('/Users/gwang/temp/OfficeLenseModels/unet_docs-08.ort', 'rb').read()
    file = open('/Users/gwang/github/onnxruntime4/onnxruntime/test/testdata/ort_github_issue_4031.onnx.ort', 'rb').read()
    buffer = bytearray(file)
    inf_sess = InferenceSession.InferenceSession.GetRootAsInferenceSession(buffer, 0)
    model = inf_sess.Model()
    graph = model.Graph()
    required_ops = {}
    read_nodes_from_graph(graph, required_ops)
    print(required_ops)
