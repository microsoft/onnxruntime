import onnx
from quantization import quantize, quant_utils

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    onnx_opt_model = onnx.load(onnx_model_path)
    quantized_onnx_model = quantize.quantize(onnx_opt_model, quantization_mode=quant_utils.QuantizationMode.IntegerOps, symmetric_weight=True, force_fusions=True)
    onnx.save(quantized_onnx_model, quantized_model_path)

quantize_onnx_model("gpt2.onnx", "gpt2-quant.onnx")