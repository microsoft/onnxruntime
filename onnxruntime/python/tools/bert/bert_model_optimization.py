import onnx
import sys
import argparse
import numpy as np
from collections import deque
from onnx import ModelProto, TensorProto, numpy_helper

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

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    model = ModelProto()
    with open(args.input, "rb") as f:
        model.ParseFromString(f.read())

    bert_model = BertOnnxModel(model, args.num_heads, args.hidden_size, args.sequence_length, args.verbose)

    bert_model.fuse_layer_norm()

    # FastGelu uses approximation for Gelu.  It is faster.
    use_approximation = True
    gelu_op_name = 'Gelu' if not use_approximation else 'FastGelu'
    bert_model.fuse_gelu(gelu_op_name)

    bert_model.fuse_reshape()

    bert_model.fuse_attention()

    bert_model.fuse_embed_layer(args.input_int32)

    # Fuse Gelu and Add Bias before it.
    bert_model.fuse_add_bias_gelu()

    # Fuse SkipLayerNormalization and Add Bias before it.
    bert_model.fuse_add_bias_skip_layer_norm()

    if args.float16:
        bert_model.convert_model_float32_to_float16()

    bert_model.remove_unused_constant()

    # Use symbolic batch dimension in input and output.
    bert_model.update_dynamic_batch_io()

    print("opset verion", bert_model.model.opset_import[0].version)

    with open(args.output, "wb") as out:
        out.write(bert_model.model.SerializeToString())

if __name__ == "__main__":
    main()
