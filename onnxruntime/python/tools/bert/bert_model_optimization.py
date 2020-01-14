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

    parser.add_argument('--tensor_flow', required=False, action='store_true')
    parser.set_defaults(tensor_flow=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    model = ModelProto()
    with open(args.input, "rb") as f:
        model.ParseFromString(f.read())

    if args.tensor_flow:
        bert_model = BertOnnxModelTF(model, args.num_heads, args.hidden_size, args.sequence_length, args.input_int32, args.float16, args.gpu_only, args.verbose)
    else:
        bert_model = BertOnnxModel(model, args.num_heads, args.hidden_size, args.sequence_length, args.input_int32, args.float16, args.gpu_only, args.verbose)

    bert_model.optimize()

    with open(args.output, "wb") as out:
        out.write(bert_model.model.SerializeToString())

if __name__ == "__main__":
    main()
