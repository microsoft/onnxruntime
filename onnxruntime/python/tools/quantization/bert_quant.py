import onnx
import numpy
from quantize import quantize, QuantizationMode

onnx_opt_model_path = "/home/yufeng/project/quantization/bert/pytorch/bert.opt.onnx"
quantized_onnx_opt_model_path_u8u8 = "/home/yufeng/project/quantization/bert/pytorch/bert.opt.u8u8.full.onnx"
quantized_onnx_opt_model_path_u8s8 = "/home/yufeng/project/quantization/bert/pytorch/bert.opt.u8s8.full.onnx"

onnx_opt_model = onnx.load(onnx_opt_model_path)
quantized_opt_onnx_model = quantize(onnx_opt_model, quantization_mode=QuantizationMode.IntegerOps, symmetric_weight=True, force_fusions=True)
onnx.save(quantized_opt_onnx_model, quantized_onnx_opt_model_path_u8s8)

onnx_opt_model = onnx.load(onnx_opt_model_path)
quantized_opt_onnx_model_u8u8 = quantize(onnx_opt_model, quantization_mode=QuantizationMode.IntegerOps, force_fusions=True)
onnx.save(quantized_opt_onnx_model_u8u8, quantized_onnx_opt_model_path_u8u8)

#onnx_opt_model_path = "/home/yufeng/project/quantization/bert/pytorch/bert.onnx"
#quantized_onnx_opt_model_path_u8u8 = "/home/yufeng/project/quantization/bert/pytorch/bert.opt.u8u8.qattention.onnx"
#quantized_onnx_opt_model_path_u8s8 = "/home/yufeng/project/quantization/bert/pytorch/bert.u8s8.onnx"
#
#onnx_opt_model = onnx.load(onnx_opt_model_path)
#quantized_opt_onnx_model = quantize(onnx_opt_model, quantization_mode=QuantizationMode.IntegerOps, symmetric_weight=True, force_fusions=True)
#onnx.save(quantized_opt_onnx_model, quantized_onnx_opt_model_path_u8s8)