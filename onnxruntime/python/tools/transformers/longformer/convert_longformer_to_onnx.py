# Before running this script, please run "python setup.py install" to build the longformer_attention.cpp
# under a python environment with PyTorch installed. Then you can update the path of longformer_attention.cpython-*.so 
# and run this script in same environment.
# Conversion tested in Ubuntu 18.04 in WSL (Windows Subsystem for Linux), python 3.6, onnxruntime 1.5.2, PyTorch 1.6.0+cpu, transformers 3.0.2
# GPU is not needed for this script. You can run it in CPU.
# For inference of the onnx model, you will need onnxruntime-gpu 1.6.0 (or nightly build).

import torch
import numpy as np
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from benchmark_longformer import get_dummy_inputs

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

is_debug = True
model_name = "longformer-random-tiny" if is_debug else "longformer-base-4096"
onnx_model_path = model_name + ".onnx"

from transformers import LongformerModel
model = LongformerModel.from_pretrained(MODELS[model_name]) # pretrained model name or directory

input_ids, attention_mask, global_attention_mask = get_dummy_inputs(
  sequence_length=7 if is_debug else 4096,
  num_global_tokens=2 if is_debug else 32, 
  device=torch.device('cpu'))

example_outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)

# A new function to replace LongformerSelfAttention.forward
def my_longformer_attention_forward(
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
from transformers.modeling_longformer import LongformerSelfAttention
#original_forward = LongformerSelfAttention.forward
LongformerSelfAttention.forward = my_longformer_attention_forward

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

# Restore Huggingface implementaiton like the following:
# LongformerSelfAttention.forward = original_forward