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
import coloredlogs
import onnx
import os
import sys
import argparse
import numpy as np
from collections import deque
from onnx import ModelProto, TensorProto, numpy_helper, load_model
from BertOnnxModel import BertOnnxModel, BertOptimizationOptions
from BertOnnxModelTF import BertOnnxModelTF
from BertOnnxModelKeras import BertOnnxModelKeras
from Gpt2OnnxModel import Gpt2OnnxModel

logger = logging.getLogger('')

# Map model type to tuple: optimizer class, export tools (pytorch, tf2onnx, keras2onnx) and whether OnnxRuntime has the optimization.
MODEL_CLASSES = {
    "bert": (BertOnnxModel, "pytorch", True),
    "bert_tf": (BertOnnxModelTF, "tf2onnx", False),
    "bert_keras": (BertOnnxModelKeras, "keras2onnx", False),
    "gpt2": (Gpt2OnnxModel, "pytorch", True)
}


def optimize_by_onnxruntime(onnx_model_path, use_gpu=False, optimized_model_path=None, opt_level=99):
    """
    Use onnxruntime package to optimize model. It could support models exported by PyTorch.

    Args:
        onnx_model_path (str): th path of input onnx model.
        use_gpu (bool): whether the optimized model is targeted to run in GPU.
        optimized_model_path (str or None): the path of optimized model.

    Returns:
        optimized_model_path: the path of optimized model
    """
    import onnxruntime

    if use_gpu and 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        logger.error("There is no gpu for onnxruntime to do optimization.")
        return onnx_model_path

    sess_options = onnxruntime.SessionOptions()
    if opt_level == 1:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt_level == 2:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        assert opt_level == 99
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    if optimized_model_path is None:
        path_prefix = onnx_model_path[:-5]  #remove .onnx suffix
        optimized_model_path = "{}_o{}_{}.onnx".format(path_prefix, opt_level, "gpu" if use_gpu else "cpu")

    sess_options.optimized_model_filepath = optimized_model_path

    if not use_gpu:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=['CPUExecutionProvider'])
    else:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options)
        assert 'CUDAExecutionProvider' in session.get_providers()  # Make sure there is GPU

    assert os.path.exists(optimized_model_path) and os.path.isfile(optimized_model_path)
    logger.info("Save optimized model by onnxruntime to {}".format(optimized_model_path))
    return optimized_model_path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help="input onnx model path")

    parser.add_argument('--output', required=True, type=str, help="optimized onnx model path")

    parser.add_argument('--model_type',
                        required=False,
                        type=str.lower,
                        default="bert",
                        choices=list(MODEL_CLASSES.keys()),
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--num_heads',
                        required=False,
                        type=int,
                        default=12,
                        help="number of attention heads. 12 for bert-base model and 16 for bert-large")

    parser.add_argument('--hidden_size',
                        required=False,
                        type=int,
                        default=768,
                        help="bert model hidden size. 768 for bert-base model and 1024 for bert-large")

    parser.add_argument('--input_int32',
                        required=False,
                        action='store_true',
                        help="Use int32 (instead of int64) tensor as input to avoid unnecessary data cast")
    parser.set_defaults(input_int32=False)

    parser.add_argument(
        '--float16',
        required=False,
        action='store_true',
        help="If your target device is V100 or T4 GPU, use this to convert float32 to float16 for best performance")
    parser.set_defaults(float16=False)

    parser.add_argument('--disable_attention', required=False, action='store_true', help="disable Attention fusion")
    parser.set_defaults(disable_attention=False)

    parser.add_argument('--disable_skip_layer_norm',
                        required=False,
                        action='store_true',
                        help="disable SkipLayerNormalization fusion")
    parser.set_defaults(disable_skip_layer_norm=False)

    parser.add_argument('--disable_embed_layer_norm',
                        required=False,
                        action='store_true',
                        help="disable EmbedLayerNormalization fusion")
    parser.set_defaults(disable_embed_layer_norm=False)

    parser.add_argument('--disable_bias_skip_layer_norm',
                        required=False,
                        action='store_true',
                        help="disable Add Bias and SkipLayerNormalization fusion")
    parser.set_defaults(disable_bias_skip_layer_norm=False)

    parser.add_argument('--disable_bias_gelu',
                        required=False,
                        action='store_true',
                        help="disable Add Bias and Gelu/FastGelu fusion")
    parser.set_defaults(disable_bias_gelu=False)

    parser.add_argument('--disable_layer_norm',
                        required=False,
                        action='store_true',
                        help="disable LayerNormalization fusion")
    parser.set_defaults(disable_layer_norm=False)

    parser.add_argument('--disable_gelu', required=False, action='store_true', help="disable Gelu fusion")
    parser.set_defaults(disable_gelu=False)

    parser.add_argument('--enable_gelu_approximation',
                        required=False,
                        action='store_true',
                        help="enable Gelu/BiasGelu to FastGelu conversion")
    parser.set_defaults(enable_gelu_approximation=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--only_onnxruntime', required=False, action='store_true', help="optimized by onnxruntime only")
    parser.set_defaults(only_onnxruntime=False)

    parser.add_argument('--opt_level',
                        required=False,
                        type=int,
                        choices=[0, 1, 2, 99],
                        default=0,
                        help="onnxruntime optimization level. 0 will disable onnxruntime.")

    args = parser.parse_args()

    return args


def get_optimization_options(args):
    optimization_options = BertOptimizationOptions(args.model_type)
    if args.disable_gelu:
        optimization_options.enable_gelu = False
    if args.disable_layer_norm:
        optimization_options.enable_layer_norm = False
    if args.disable_attention:
        optimization_options.enable_attention = False
    if args.disable_skip_layer_norm:
        optimization_options.enable_skip_layer_norm = False
    if args.disable_embed_layer_norm:
        optimization_options.enable_embed_layer_norm = False
    if args.disable_bias_skip_layer_norm:
        optimization_options.enable_bias_skip_layer_norm = False
    if args.disable_bias_gelu:
        optimization_options.enable_bias_gelu = False
    if args.enable_gelu_approximation:
        optimization_options.enable_gelu_approximation = True
    return optimization_options


def optimize_model(input,
                   model_type,
                   num_heads,
                   hidden_size,
                   opt_level=0,
                   optimization_options=None,
                   use_gpu=False,
                   only_onnxruntime=False):
    (optimizer_class, producer, run_onnxruntime) = MODEL_CLASSES[model_type]

    input_model_path = input

    if opt_level > 1: # Optimization specified for an execution provider.
        input_model_path = optimize_by_onnxruntime(input_model_path, use_gpu=use_gpu, opt_level=opt_level)
    elif run_onnxruntime:
        # Use Onnxruntime to do optimizations (like constant folding and cast elimation) that is not specified to exection provider.
        # CPU provider is used here so that there is no extra node for GPU memory copy.
        input_model_path = optimize_by_onnxruntime(input_model_path, use_gpu=False, opt_level=1)

    model = load_model(input_model_path, format=None, load_external_data=True)

    if model.producer_name and producer != model.producer_name:
        logger.warning(
            f"Model producer not matched: Expect {producer},  Got {model.producer_name} {model.producer_version}. Please specify correct --model_type parameter."
        )

    if optimization_options is None:
        optimization_options = BertOptimizationOptions(model_type)

    bert_model = optimizer_class(model, num_heads, hidden_size)

    if not only_onnxruntime:
        bert_model.optimize(optimization_options)

    return bert_model


def setup_logger(verbose):
    if verbose:
        coloredlogs.install(level='DEBUG', fmt='[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s')
    else:
        coloredlogs.install(fmt='%(funcName)20s: %(message)s')


def main():
    args = parse_arguments()

    setup_logger(args.verbose)

    optimization_options = get_optimization_options(args)

    bert_model = optimize_model(args.input,
                                args.model_type,
                                args.num_heads,
                                args.hidden_size,
                                opt_level=args.opt_level,
                                optimization_options=optimization_options,
                                use_gpu=args.use_gpu,
                                only_onnxruntime=args.only_onnxruntime)

    if args.float16:
        bert_model.convert_model_float32_to_float16()

    if args.input_int32:
        bert_model.change_input_to_int32()

    bert_model.save_model_to_file(args.output)

    if bert_model.is_fully_optimized():
        logger.info("The output model is fully optimized.")
    else:
        logger.warning("The output model is not fully optimized. It might not be usable.")


if __name__ == "__main__":
    main()
