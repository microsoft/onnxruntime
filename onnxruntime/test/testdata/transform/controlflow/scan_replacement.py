import onnx
from onnx import helper, TensorProto

add = helper.make_node("Add", ["y_last", "x"], ["y"])
identity = helper.make_node("Identity", ["y"], ["y_next"])

tinfo = lambda shape: helper.make_tensor_type_proto(TensorProto.FLOAT, shape)
vinfo = lambda x, shape: helper.make_value_info(x, tinfo(shape))
subgraph = helper.make_graph([add, identity], "scan-subgraph",
                          [vinfo("y_last", []), vinfo("x", [])],
                          [vinfo("y_next", []), vinfo("y", [])])

scan = helper.make_node("Scan", ["y_last", "x"], ["y_next", "y"], "scan", body=subgraph, num_scan_inputs=1)

graph = helper.make_graph(
        [scan], "model", [vinfo("y_last", []), vinfo("x", ["seq"])], [vinfo("y_next", []), vinfo("y", ["seq"])])

model = helper.make_model(graph)

onnx.checker.check_model(model, full_check=True)

onnx.save_model(model, "scan_replacement.onnx")
