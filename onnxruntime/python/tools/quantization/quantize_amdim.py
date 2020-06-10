import onnx
import onnxruntime

from quantize import quantize, QuantizationMode

model = onnx.load('D:\\projects\\quantiazation\\amdim\\model_518\\model_lite.onnx')
#nodes = model.graph.node
#index = 0
#for node in nodes:
#    node.name = str(index)
#    index= index+1

print('quantizing...')
quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps, force_fusions=False)

print('saving....')
onnx.save(quantized_model, r'D:\projects\quantiazation\amdim\model_518\model_lite.quant.onnx')
