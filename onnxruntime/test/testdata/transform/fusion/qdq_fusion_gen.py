import onnx
from onnx import OperatorSetIdProto, TensorProto, helper

# inputs/outputs
A = helper.make_tensor_value_info("A", TensorProto.FLOAT, ["unk_1", "unk_2", 1024, 4096])
B = helper.make_tensor_value_info("B", TensorProto.FLOAT, ["unk_1", "unk_2", 1024, 4096])

# initializers
quant_scale = helper.make_tensor("quant_scale", TensorProto.FLOAT, [], [0.075])
quant_zero_point_int = helper.make_tensor("quant_zero_point", TensorProto.INT8, [], [20])
quant_zero_point_uint = helper.make_tensor("quant_zero_point", TensorProto.UINT8, [], [120])

# opsets
opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = ""
opsets.append(onnxdomain)
kwargs = {}
kwargs["opset_imports"] = opsets

# graph nodes
quantize_linear = helper.make_node(
    "QuantizeLinear", ["A", "quant_scale", "quant_zero_point"], ["quantized_value"], "quanztize0"
)
dequantize_linear = helper.make_node(
    "DequantizeLinear", ["quantized_value", "quant_scale", "quant_zero_point"], ["B"], "dequanztize0"
)
quantize_linear_zp_not_provided = helper.make_node(
    "QuantizeLinear", ["A", "quant_scale"], ["quantized_value"], "quanztize0"
)
dequantize_linear_zp_not_provided = helper.make_node(
    "DequantizeLinear", ["quantized_value", "quant_scale"], ["B"], "dequanztize0"
)

# graph with int8 quantization
graph = helper.make_graph(
    [quantize_linear, dequantize_linear],
    "QDQFusion",
    [A],
    [B],
    [quant_scale, quant_zero_point_int],
)

# model with int8 quantization
model = helper.make_model(graph, producer_name="onnx-example", **kwargs)
onnx.save(model, "qdq_fusion_int8.onnx")

# graph with uint8 quantization
graph = helper.make_graph(
    [quantize_linear, dequantize_linear],
    "QDQFusion",
    [A],
    [B],
    [quant_scale, quant_zero_point_uint],
)

# model with uint8 quantization
model = helper.make_model(graph, producer_name="onnx-example", **kwargs)
onnx.save(model, "qdq_fusion_uint8.onnx")

# graph with no zero point initializer
graph = helper.make_graph(
    [quantize_linear_zp_not_provided, dequantize_linear_zp_not_provided],
    "QDQFusion",
    [A],
    [B],
    [quant_scale],
)

# model with no zero point initializer
model = helper.make_model(graph, producer_name="onnx-example", **kwargs)
onnx.save(model, "qdq_fusion_zp_not_provided.onnx")
