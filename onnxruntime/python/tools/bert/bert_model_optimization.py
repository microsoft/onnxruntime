#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# Convert Bert ONNX model converted from TensorFlow or exported from PyTorch to use Attention, Gelu,
# SkipLayerNormalization and EmbedLayerNormalization ops to optimize
# performance on NVidia GPU and CPU.

# Note: This script is not required for Bert model exported from PyTorch. 
# OnnxRuntime has bert model optimization support internally. The recommended way is
# to set optimization level to ORT_ENABLE_EXTENDED during Bert model inference.
# See the following document for more information:
# https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Graph_Optimizations.md

# This script is retained for experiment purpose. Useful senarios like the following:
#  (1) Change model from fp32 to fp16.
#  (2) Change input data type from int64 to int32.
#  (3) Model cannot be handled to OnnxRuntime graph optimization, and you can modify this script to get optimized model.



import onnx
import sys
import argparse
import numpy as np
from collections import deque
from onnx import ModelProto, TensorProto, numpy_helper
from BertOnnxModel import BertOnnxModel
from BertOnnxModelTF import BertOnnxModelTF

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--framework', required=True, type=str, help="Original framework. Only support TensorFlow and PyTorch")

    # model parameters
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

    args = parser.parse_args()

    model = ModelProto()
    with open(args.input, "rb") as f:
        model.ParseFromString(f.read())

    if args.framework.lower() == 'tensorflow':
        bert_model = BertOnnxModelTF(model, args.num_heads, args.hidden_size, args.sequence_length, args.input_int32, args.float16, args.gpu_only, args.verbose)
    elif args.framework.lower() == 'pytorch':
        bert_model = BertOnnxModel(model, args.num_heads, args.hidden_size, args.sequence_length, args.input_int32, args.float16, args.gpu_only, args.verbose)
    else:
        print("Unsupported framework:" + args.framework)

    bert_model.optimize()

    with open(args.output, "wb") as out:
        out.write(bert_model.model.SerializeToString())

if __name__ == "__main__":
    main()
