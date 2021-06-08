import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum
import onnxruntime
from onnxruntime import quantization

#
# TODO(kreeger): This script should just quantize the existing models.
#
MODEL_LIST = [
    "embed_layer_norm_format3.onnx"
]

for model_name in MODEL_LIST:
    # First, optimize the model with onnxruntime:
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    
    path_prefix = model_name[:-5]  # remove .onnx suffix
    optimized_model_path = "{}_opt.onnx".format(path_prefix)
    sess_options.optimized_model_filepath = optimized_model_path
    session = onnxruntime.InferenceSession(model_name, sess_options)

    onnxruntime.quantization.quantize_dynamic(optimized_model_path, "{}_opt.quant.onnx".format(path_prefix))
