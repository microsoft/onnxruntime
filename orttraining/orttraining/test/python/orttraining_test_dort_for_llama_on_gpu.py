import time
import torch
import onnxscript
import onnxruntime
from onnxruntime.training.experimental import (
  _modeling_llama,
  _transformers,
)
from onnxruntime.training.ortmodule.torch_cpp_extensions.cpu.aten_op_executor import load_aten_op_executor_cpp_extension
load_aten_op_executor_cpp_extension()
from transformers import LlamaConfig  # noqa: F811
from transformers.models.llama.modeling_llama import LlamaModel  # noqa: F811
from torch import optim

onnx_transformer_backend = _transformers.make_onnxrt_transformer_backend()


device = "cuda"

# config = LlamaConfig(
#    num_hidden_layers=1
#    vocab_size=1024,
#    hidden_size=16,
#    intermediate_size=16,
#    max_position_embeddings=256,
#    num_attention_heads=2,
#    hidden_dropout_prob=0.0,
#    attention_dropout_prob=0.0,
# )

config = LlamaConfig(num_hidden_layers=1)

#config._attn_implementation = "eager"
config._attn_implementation = "sdpa"

class LlamaModelWrapper(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.llama = LlamaModel(config)

    def forward(self, input_ids, attention_mask, position_ids):
        decoder_output = self.llama(input_ids, attention_mask, position_ids, return_dict=False)
        return decoder_output[0]


def generate_example_inputs(batch: int, seq: int):
    # shape: batch x seq x hidden_size
    input_ids = torch.randint(0, 7, size=(batch, seq), dtype=torch.int64).to(device)
    # Usually, its shape is a tensor with shape batch x seq x seq.
    # However, to bypass some control flow in the model, we use None.
    attention_mask = None
    position_ids = torch.arange(0, seq, dtype=torch.int64).to(device)
    position_ids = position_ids.unsqueeze(0).view(-1, seq).to(device)
    return input_ids, attention_mask, position_ids


# Reason for using multiple example argument groups:
#  Export model to ONNX with one example argument group
#  and test it with other example argument groups.
example_args_collection = (
    generate_example_inputs(2, 1024),
    generate_example_inputs(2, 1024),
    generate_example_inputs(2, 1024),
    generate_example_inputs(2, 1024),
    generate_example_inputs(2, 1024),
)

model = LlamaModelWrapper(config).eval().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()

compiled_model = torch.compile(model, backend=onnx_transformer_backend, dynamic=True)

result = compiled_model(*example_args_collection[0])
target = torch.rand_like(result, memory_format=torch.contiguous_format).to(device)
loss = torch.nn.functional.mse_loss(result, target)
loss.backward()
print(loss)

#start_time = time.time()
#for i, example_inputs in enumerate(example_args_collection):
#    torch.cuda.nvtx.range_push(f"batch{i}")
#    with torch.autocast(device_type='cuda', dtype=torch.float16):
#        torch.cuda.nvtx.range_push("FW")
#        result = compiled_model(*example_args_collection[i])
#        torch.cuda.nvtx.range_pop()
#
#        torch.cuda.nvtx.range_push("Loss")
#        target = torch.rand_like(result, memory_format=torch.contiguous_format).to(device)
#        loss = torch.nn.functional.mse_loss(result, target)
#        torch.cuda.nvtx.range_pop()
#
#        torch.cuda.nvtx.range_push("BW")
#        loss.backward()
#        torch.cuda.nvtx.range_pop()
#
#        torch.cuda.nvtx.range_push("Optim")
#        optimizer.step()
#        optimizer.zero_grad()
#        torch.cuda.nvtx.range_pop()
#
#    torch.cuda.nvtx.range_pop()
#torch.cuda.synchronize()
#end_time = time.time()
#print(f"Avg time: {(end_time - start_time) / (len(example_args_collection))}")
#
#print(result)