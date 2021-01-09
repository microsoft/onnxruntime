# Before running this script, please run "python setup.py install" to build the longformer_attention.cpp
# under a python environment with PyTorch installed. Then you can update the path of longformer_attention.cpython-*.so
# and run this script in same environment.
# Conversion tested in Ubuntu 18.04 in WSL (Windows Subsystem for Linux), python 3.6, onnxruntime 1.5.2, PyTorch 1.6.0+cpu, transformers 3.0.2
# GPU is not needed for this script. You can run it in CPU.
# For inference of the onnx model, you will need onnxruntime-gpu 1.6.0 (or nightly build).

import torch
import numpy as np
import argparse
import transformers
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from packaging import version

@parse_args('v', 'v', 'v', 'v','v', 'v', 'v', 'i', 'i')
def my_longformer_attention(g, input, weight, bias, mask, global_weight, global_bias, global_mask, num_heads, window):
  return g.op("com.microsoft::LongformerAttention", input, weight, bias, mask, global_weight, global_bias, global_mask, num_heads_i=num_heads, window_i=window)

# namespace is onnxruntime which is registered in longformer_attention.cpp
register_custom_op_symbolic('onnxruntime::LongformerAttention', my_longformer_attention, 9)

# TODO: update the path according to output of "python setup.py install" when your python version is not 3.6
torch.ops.load_library(r'build/lib.linux-x86_64-3.6/longformer_attention.cpython-36m-x86_64-linux-gnu.so')

# mapping from model name to pretrained model name
MODELS = {
    "longformer-base-4096": "allenai/longformer-base-4096",
    "longformer-random-tiny": "patrickvonplaten/longformer-random-tiny"  # A tiny model for debugging
}

is_debug = False

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m",
                        "--model",
                        required=False,
                        type=str,
                        default="longformer-random-tiny" if is_debug else "longformer-base-4096",
                        choices=list(MODELS.keys()),
                        help="Pre-trained models in the list: " + ", ".join(MODELS.keys()))

    # Sequence length shall choose properly.
    # If multiple of windows size is used, there is no padding in ONNX model so you will need padding by yourself before running onnx model.
    parser.add_argument("-s",
                        "--sequence_length",
                        type=int,
                        default=4 if is_debug else 512)

    parser.add_argument("-g", "--global_length", type=int, default=1 if is_debug else 8)

    parser.add_argument('-o',
                        '--optimize_onnx',
                        required=False,
                        action='store_true',
                        help='Use optimizer.py to optimize onnx model')
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=str,
        default='fp32',
        choices=['fp32', 'fp16'],
        help="Precision of model to run: fp32 for full precision, fp16 for mixed precision")

    args = parser.parse_args()
    return args

def get_dummy_inputs(sequence_length, num_global_tokens, device):
    # Create dummy inputs
    input_ids = torch.arange(sequence_length).unsqueeze(0).to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long,
                                device=input_ids.device)  # TODO: use random word ID. #TODO: simulate masked word
    global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
    if num_global_tokens > 0:
        global_token_index = list(range(num_global_tokens))
        global_attention_mask[:, global_token_index] = 1
    # TODO: support more inputs like token_type_ids, position_ids
    return input_ids, attention_mask, global_attention_mask

args = parse_arguments()

model_name = args.model
onnx_model_path = model_name + ".onnx"

from transformers import LongformerModel
model = LongformerModel.from_pretrained(MODELS[model_name]) # pretrained model name or directory

input_ids, attention_mask, global_attention_mask = get_dummy_inputs(sequence_length=args.sequence_length, num_global_tokens=args.global_length, device=torch.device('cpu'))

example_outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)

# A new function to replace LongformerSelfAttention.forward
#For transformers 4.0
def my_longformer_self_attention_forward_4(self, hidden_states, attention_mask=None, is_index_masked=None, is_index_global_attn=None, is_global_attn=None):
    # TODO: move mask calculation to LongFormerModel class to avoid calculating it again and again in each layer.
    global_mask = is_index_global_attn.int()
    torch.masked_fill(attention_mask, is_index_global_attn, 0.0)

    weight = torch.stack((self.query.weight.transpose(0,1), self.key.weight.transpose(0,1), self.value.weight.transpose(0,1)), dim=1)
    weight = weight.reshape(self.embed_dim, 3*self.embed_dim)

    bias = torch.stack((self.query.bias, self.key.bias, self.value.bias), dim=0)
    bias = bias.reshape(3 * self.embed_dim)

    global_weight = torch.stack((self.query_global.weight.transpose(0,1), self.key_global.weight.transpose(0,1), self.value_global.weight.transpose(0,1)), dim=1)
    global_weight = global_weight.reshape(self.embed_dim, 3*self.embed_dim)

    global_bias = torch.stack((self.query_global.bias, self.key_global.bias, self.value_global.bias), dim=0)
    global_bias = global_bias.reshape(3 * self.embed_dim)

    attn_output = torch.ops.onnxruntime.LongformerAttention(hidden_states, weight, bias, attention_mask, global_weight, global_bias, global_mask, self.num_heads, self.one_sided_attn_window_size)

    assert attn_output.size() == hidden_states.size(), "Unexpected size"

    outputs = (attn_output,)
    return outputs


# For transformers 3.0
def my_longformer_attention_forward_3(
    self,
    hidden_states,
    attention_mask,
    output_attentions=False):

    assert output_attentions is False

    # TODO: attention_mask can directly be passed from inputs of model to avoid these processing.
    attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
    global_mask = (attention_mask > 0).int()
    torch.masked_fill(attention_mask, attention_mask > 0, 0.0)

    weight = torch.stack((self.query.weight.transpose(0,1), self.key.weight.transpose(0,1), self.value.weight.transpose(0,1)), dim=1)
    weight = weight.reshape(self.embed_dim, 3*self.embed_dim)

    bias = torch.stack((self.query.bias, self.key.bias, self.value.bias), dim=0)
    bias = bias.reshape(3 * self.embed_dim)

    global_weight = torch.stack((self.query_global.weight.transpose(0,1), self.key_global.weight.transpose(0,1), self.value_global.weight.transpose(0,1)), dim=1)
    global_weight = global_weight.reshape(self.embed_dim, 3*self.embed_dim)

    global_bias = torch.stack((self.query_global.bias, self.key_global.bias, self.value_global.bias), dim=0)
    global_bias = global_bias.reshape(3 * self.embed_dim)

    attn_output = torch.ops.onnxruntime.LongformerAttention(hidden_states, weight, bias, attention_mask, global_weight, global_bias, global_mask, self.num_heads, self.one_sided_attn_window_size)

    assert attn_output.size() == hidden_states.size(), "Unexpected size"

    outputs = (attn_output,)
    return outputs

# Here we replace LongformerSelfAttention.forward using our implmentation for exporting ONNX model
if version.parse(transformers.__version__) < version.parse("4.0.0"):
    from transformers.modeling_longformer import LongformerSelfAttention
    #original_forward = LongformerSelfAttention.forward
    LongformerSelfAttention.forward = my_longformer_attention_forward_3
else:
    from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
    #original_forward = LongformerSelfAttention.forward
    LongformerSelfAttention.forward = my_longformer_self_attention_forward_4

# TODO: support more inputs like (input_ids, attention_mask, global_attention_mask, token_type_ids, position_ids)
example_inputs = (input_ids, attention_mask, global_attention_mask)

torch.onnx.export(model, example_inputs, onnx_model_path,
                  opset_version=11,
                  example_outputs=example_outputs,
                  input_names=["input_ids", "attention_mask", "global_attention_mask"], output_names=["last_state", "pooler"],
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                                'global_attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                                'last_state': {0: 'batch_size', 1: 'sequence_length'},
                                'pooler': {0: 'batch_size', 1: 'sequence_length'}
                                },
                  custom_opsets={"com.microsoft": 1}
                  )
print(f"ONNX model exported to {onnx_model_path}")

if args.sequence_length % model.config.attention_window[0] == 0:
    print(f"*Attention*: You need input padding for inference: input sequece length shall be multiple of {model.config.attention_window[0]}. It is because the example input for export ONNX model does not need padding so padding logic is not in onnx model.")

# Restore Huggingface implementaiton like the following:
# LongformerSelfAttention.forward = original_forward

if args.precision != 'fp32' or args.optimize_onnx:
    from onnx import load_model
    from onnxruntime.transformers.onnx_model_bert import BertOnnxModel, BertOptimizationOptions
    model = load_model(onnx_model_path, format=None, load_external_data=True)
    optimization_options = BertOptimizationOptions('bert')
    optimizer = BertOnnxModel(model, num_heads=16, hidden_size=768)
    optimizer.optimize(optimization_options)
    optimized_model_path = model_name + "_fp32.onnx"
    optimizer.save_model_to_file(optimized_model_path)
    print(f"optimized fp32 model saved to {optimized_model_path}")

    if args.precision == 'fp16':
        optimizer.convert_model_float32_to_float16(cast_input_output=True)
        optimized_model_path = model_name + "_fp16.onnx"
        optimizer.save_model_to_file(optimized_model_path)
        print(f"optimized fp16 model saved to {optimized_model_path}")
