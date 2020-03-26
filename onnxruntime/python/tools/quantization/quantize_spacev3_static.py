import onnx
import numpy as np
from quantize import quantize, QuantizationMode


def calibrate_custom(model):
    quant_dict = {}
    for node in model.graph.node:
        if node.op_type == "MatMul" or node.op_type == 'Attention':
            quant_dict[node.input[0]] = [np.int8(0), np.float32(0.047)]
    return quant_dict


model = onnx.load('D:\projects\quantiazation\spacev3\spacev3.opt.fp32.onnx')

quant_dict = calibrate_custom(model)

quantized_model = quantize(model,
                           quantization_mode=QuantizationMode.IntegerOps,
                           static=True,
                           force_fusions=False,
                           symmetric_activation=True,
                           symmetric_weight=True,
                           quantization_params=quant_dict)
onnx.save(
    quantized_model,
    'D:\projects\quantiazation\spacev3\spacev3.opt.fp32.quant.attention.static.onnx')