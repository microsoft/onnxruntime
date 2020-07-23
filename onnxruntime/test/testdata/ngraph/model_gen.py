import numpy as np
import onnx
from onnx import helper, numpy_helper


def save_graph(graph_def, name):
    model_def = helper.make_model(graph_def, producer_name="nGraph EP test model")
    model_def.opset_import[0].version = 7
    model_def.ir_version = 3
    onnx.save_model(model_def, name)


A = helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, [4])
B = helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [4])
C = helper.make_tensor_value_info('C', onnx.TensorProto.FLOAT, [4])
Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [4])
Z = helper.make_tensor_value_info('Z', onnx.TensorProto.FLOAT, [4])

graph_def = helper.make_graph([
    helper.make_node('Add', ['A', 'A'], ['node1_out'], "node_1"),
    helper.make_node('Mul', ['node1_out', 'B'], ['Z'], "node_2")
], "nGraph EP test graph", [A, B], [Z])

save_graph(graph_def, "Basic_Test.onnx")

graph_def = helper.make_graph([
    helper.make_node('Add', ['A', 'A'], ['node1_out'], "node_1"),
    helper.make_node('Mul', ['node1_out', 'B'], ['node2_out'], "node_2"),
    helper.make_node('UnSupportedOp', ['node2_out'], ['Z'], "node_3")
], "nGraph EP test graph", [A, B], [Z])

save_graph(graph_def, "Graph_with_UnSupportedOp.onnx")

graph_def = helper.make_graph([
    helper.make_node('Add', ['A', 'A'], ['node1_out'], "node_1"),
    helper.make_node('Mul', ['node1_out', 'B'], ['node2_out'], "node_2"),
    helper.make_node('UnSupportedOp', ['node2_out'], ['node3_out'], "node_3"),
    helper.make_node('Add', ['node3_out', 'C'], ['Z'], "node_4")
], "nGraph EP test graph", [A, B, C], [Z])

save_graph(graph_def, "Two_Subgraphs.onnx")

graph_def = helper.make_graph([
    helper.make_node('Add', ['A', 'A'], ['Y'], "node_1"),
    helper.make_node('Mul', ['Y', 'B'], ['node2_out'], "node_2"),
    helper.make_node('UnSupportedOp', ['node2_out'], ['node3_out'], "node_3"),
    helper.make_node('Add', ['node3_out', 'C'], ['Z'], "node_4")
], "nGraph EP test graph", [A, B, C], [Y, Z])

save_graph(graph_def, "ClusterOut_isAlso_GraphOut.onnx")

graph_def = helper.make_graph([
    helper.make_node('Add', ['A', 'A'], ['node1_out'], "node_1"),
    helper.make_node('Mul', ['node1_out', 'B'], ['Y'], "node_2"),
    helper.make_node('UnSupportedOp', ['Y'], ['node3_out'], "node_3"),
    helper.make_node('Add', ['node3_out', 'C'], ['Z'], "node_4")
], "nGraph EP test graph", [A, B, C], [Y, Z])

save_graph(graph_def, "InOut_isAlso_GraphOut.onnx")

graph_def = helper.make_graph([
    helper.make_node('Add', ['A', 'A'], ['node1_out'], "node_1"),
    helper.make_node('Dropout', ['node1_out'], ['Y', 'X'], "node_2"),
    helper.make_node('UnSupportedOp', ['Y'], ['node3_out'], "node_3"),
    helper.make_node('Add', ['node3_out', 'C'], ['Z'], "node_4")
], "nGraph EP test graph", [A, C], [Z])

save_graph(graph_def, "Op_with_Optional_or_Unused_Outputs.onnx")

one_in = helper.make_tensor_value_info('one', onnx.TensorProto.FLOAT, [1])
one_data = np.array([1], dtype=np.float32)
one_tensor = numpy_helper.from_array(one_data, "one")

graph_def = helper.make_graph([
    helper.make_node('Add', ['A', "A"], ['node1_out'], "node_1"),
    helper.make_node('UnSupportedOp', ['node1_out'], ['branch_a_out'], "node_2"),
    helper.make_node('Add', ["node1_out", "one"], ["branch_b_out1"], "node_3"),
    helper.make_node('Add', ["branch_b_out1", "one"], ["branch_b_out"], "node_4"),
    helper.make_node('Add', ["branch_a_out", "branch_b_out"], ["Z"], "node_5")
], "nGraph EP test graph", [A, one_in], [Z], [one_tensor])

save_graph(graph_def, "Independent_SubGraphs.onnx")
