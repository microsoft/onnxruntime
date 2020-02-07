#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# Convert Bert ONNX model converted from TensorFlow or exported from PyTorch to use Attention, Gelu,
# SkipLayerNormalization and EmbedLayerNormalization ops to optimize
# performance on NVidia GPU and CPU.

# For Bert model exported from PyTorch, OnnxRuntime has bert model optimization support internally.
# You can use the option --use_onnxruntime to use model optimization from OnnxRuntime package.
# For Bert model file like name.onnx, optimized model for GPU or CPU from OnnxRuntime will output as
# name_ort_gpu.onnx or name_ort_cpu.onnx in the same directory.
# This script is retained for experiment purpose. Useful senarios like the following:
#  (1) Change model from fp32 to fp16.
#  (2) Change input data type from int64 to int32.
#  (3) Some model cannot be handled by OnnxRuntime, and you can modify this script to get optimized model.

# This script has been tested using the following models:
#  (1) BertForSequenceClassification as in https://github.com/huggingface/transformers/blob/master/examples/run_glue.py
#      PyTorch 1.2 or above, and exported to Onnx using opset version 10 or 11.
#  (2) BertForQuestionAnswering as in https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
#      PyTorch 1.2 or above, and exported to Onnx using opset version 10 or 11.

import logging
import onnx
import os
import sys
import argparse
import numpy as np
from collections import deque
from onnx import ModelProto, TensorProto, numpy_helper
import onnxruntime
from BertOnnxModel import BertOnnxModel
from BertOnnxModelTF import BertOnnxModelTF

logger = logging.getLogger('')

def run_onnxruntime(onnx_model_path, use_gpu, optimized_model_path=None):
    if use_gpu and 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        logger.error("There is no gpu for onnxruntime to do optimization.")

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    if optimized_model_path is None:
        path_prefix = onnx_model_path[:-5]   #remove .onnx suffix
        optimized_model_path = "{}_ort_{}.onnx".format(path_prefix, "gpu" if use_gpu else "cpu")

    sess_options.optimized_model_filepath = optimized_model_path

    if not use_gpu:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=['CPUExecutionProvider'])
    else:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options)
        assert 'CUDAExecutionProvider' in session.get_providers() # Make sure there is GPU

    assert os.path.exists(optimized_model_path) and os.path.isfile(optimized_model_path)
    logger.info("Save optimized model by onnxruntime to {}".format(optimized_model_path))
    return optimized_model_path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--framework', required=True, type=str.lower, choices=["tensorflow", "pytorch"], help="Original framework")

    # model parameters.
    parser.add_argument('--num_heads', required=False, type=int, default=12, help="number of attention heads")
    parser.add_argument('--hidden_size', required=False, type=int, default=768)
    parser.add_argument('--sequence_length', required=False, type=int, default=128)

    # Use int32 (instead of int64) tensor as input to avoid unnecessary data
    # type cast.
    parser.add_argument('--input_int32', required=False, action='store_true')
    parser.set_defaults(input_int32=False)

    # For NVidia GPU with Tensor Core like V100 and T4, half-precision float
    # brings better performance.
    parser.add_argument('--float16', required=False, action='store_true')
    parser.set_defaults(float16=False)

    parser.add_argument('--gpu_only', required=False, action='store_true')
    parser.set_defaults(gpu_only=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('--use_onnxruntime', required=False, action='store_true')
    parser.set_defaults(use_onnxruntime=False)

    args = parser.parse_args()

    return args

def optimize_model(input, framework, gpu_only, num_heads, hidden_size, sequence_length, input_int32, float16, verbose):
    model = ModelProto()
    with open(input, "rb") as f:
        model.ParseFromString(f.read())

    if framework == 'tensorflow':
        bert_model = BertOnnxModelTF(model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only, verbose)
    else: #framework == 'pytorch'
        bert_model = BertOnnxModel(model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only, verbose)

    bert_model.optimize()
    
    return bert_model

def output_model(model, output):
    if output.endswith(".json"): # output to JSON. Only for test purpose.
        if isinstance(model, ModelProto):
            with open(output, "w") as out:
                out.write(str(model))
            logger.info("Output JSON to {}.json".format(output))
    else:
        with open(output, "wb") as out:
            out.write(model.SerializeToString())
        logger.info("Output final model to {}".format(output))

def main():
    args = parse_arguments()

    # output logging to stdout
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    log_handler.setLevel(logging.DEBUG)
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)

    input_model_path = args.input
    if args.use_onnxruntime:
        if framework == 'tensorflow':
            logger.warning("onnxruntime does not have optimization for tensorflow model. Ignore the option --use_onnxruntime.")
        else:
            input_model_path = run_onnxruntime(input_model_path, args.gpu_only)

    bert_model = optimize_model(input_model_path, args.framework, args.gpu_only, args.num_heads, args.hidden_size, args.sequence_length, args.input_int32, args.float16, args.verbose)

    output_model(bert_model.model, args.output)



if __name__ == "__main__":
    main()
