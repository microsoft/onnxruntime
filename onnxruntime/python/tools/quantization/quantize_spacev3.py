import onnx
from quantize import quantize, QuantizationMode
print('111111')
model = onnx.load('D:\projects\quantiazation\spacev3\spacev3.opt.fp32.onnx')

quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps, force_fusions=False, symmetric_activation=True, symmetric_weight = True)
onnx.save(quantized_model, 'D:\projects\quantiazation\spacev3\spacev3.opt.fp32.quant.attention.onnx')