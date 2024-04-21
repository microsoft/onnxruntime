"""Inference-only Mixtral model."""
import unittest

import numpy as np
from mpi4py import MPI
from onnx import TensorProto, helper
import torch
import time

import onnxruntime as ort

np.random.seed(3)

comm = MPI.COMM_WORLD


def get_rank():
    return comm.Get_rank()


def get_size():
    return comm.Get_size()


def print_out(*args):
    if get_rank() == 0:
        print(*args)


rank = get_rank()
tensor_parallel_size = get_size()

input_ids_name = "input_ids"
attn_mask_name = "attention_mask"
#pos_ids_name = "position_ids"

logits_name = "logits"
past_name = "past_key_values"
present_name = "present"


pt_to_np = {
    "torch.int32": np.int32,
    "torch.int64": np.int64,
    "torch.float32": np.float32,
    "torch.float16": np.float16
}

max_sequence_length = 1024 + 16 # subject to change

def get_initial_inputs_and_outputs_for_bench(batch_size, sequence_length, tensor_parallel_size, device: torch.device, use_fp16: bool, use_buffer_share: bool):
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    input_ids = torch.randint(0, 32000, (batch_size, sequence_length), device=device, dtype=torch.int64)
    attention_mask = torch.ones(batch_size, sequence_length, device=device, dtype=torch.int64)
    #position_ids = torch.arange(0, sequence_length, device=device, dtype=torch.int64).repeat(batch_size, 1)

    inputs = {
        input_ids_name: input_ids.contiguous(),
        attn_mask_name: attention_mask.contiguous(),
        #pos_ids_name: position_ids.contiguous(),
    }

    batch_size, sequence_length = input_ids.shape
    num_heads, head_size = int(8 / tensor_parallel_size), 128
    for i in range(32):
        past_key = torch.zeros(batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, device=device, dtype=torch_dtype)
        past_value = torch.zeros(batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, device=device, dtype=torch_dtype)
        inputs.update({
            past_name + f".{i}.key": past_key.contiguous(),
            past_name + f".{i}.value": past_value.contiguous(),
        })

    logits = torch.zeros(batch_size, sequence_length, 32000, device=device, dtype=torch_dtype)
    outputs = {
        logits_name: logits.contiguous()
    }

    return inputs, outputs


def apply_io_binding(model: ort.InferenceSession, inputs: dict, outputs: dict, use_fp16: bool, use_buffer_share: bool):
    # Check that all model inputs will be provided
    model_inputs = set(map(lambda model_input: model_input.name, model.get_inputs()))
    user_inputs = set(inputs.keys())
    missing_inputs = model_inputs - user_inputs
    if len(missing_inputs):
        print(f"The following model inputs are missing: {missing_inputs}")
        raise Exception("There are missing inputs to the model. Please add them and try again.")

    # Remove unnecessary inputs from model inputs
    unnecessary_inputs = user_inputs - model_inputs
    if len(unnecessary_inputs):
        for unnecessary_input in unnecessary_inputs:
            print(f"Removing unnecessary input '{unnecessary_input}' from user provided inputs")
            del inputs[unnecessary_input]

    # Bind inputs/outputs to IO binding
    io_binding = model.io_binding()
    device = None

    for k, v in inputs.items():
        io_binding.bind_input(
            name=k,
            device_type=v.device.type,
            device_id=0 if v.device.type == "cpu" else v.device.index,
            element_type=pt_to_np[repr(v.dtype)],
            shape=tuple(v.shape),
            buffer_ptr=v.data_ptr()
        )
        device = v.device

    for output in model.get_outputs():
        name = output.name
        if use_buffer_share and "present" in name:
            # Bind KV cache outputs to KV cache inputs
            v = inputs[name.replace("present", "past_key_values")]
            io_binding.bind_output(
                name=name,
                device_type=v.device.type,
                device_id=v.device.index,
                element_type=np.float16,
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr()
            )
        else:
            v = outputs[name]
            io_binding.bind_output(
                name=name,
                device_type=device.type,
                device_id=0 if device.type == "cpu" else device.index,
                element_type=(np.float16 if use_fp16 else np.float32),
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr()
            )

    return io_binding

def infer_model(tensor_parallel_size, rank, model_or_sess):
    device = torch.device(rank)
    use_fp16 = True
    use_buffer_share = True # buffer sharing is not supported now
    eos_token_id = 50256 # never let it reach eos for benchmark
    for token_num in [16]:
        for batch_size in [1]:
            for seq_len in [1024, 1024, 1024]:
                max_length = seq_len + token_num
                if rank == 0:
                    print("batch size:", batch_size, "seq len:", seq_len, "token_num:", token_num)
                # Get model and its initial inputs/outputs
                inputs, outputs = get_initial_inputs_and_outputs_for_bench(batch_size, seq_len, tensor_parallel_size, device, use_fp16, use_buffer_share)

                all_token_ids = inputs[input_ids_name].clone()
                batch_size, sequence_length = all_token_ids.shape

                current_length = sequence_length
                has_eos = torch.zeros(batch_size, device=device, dtype=torch.bool)

                count = 0
                start = time.time()
                while current_length < max_length:
                    # Run inference
                    if count == 1:
                        prompt_fence = time.time()

                    io_binding = apply_io_binding(model_or_sess, inputs, outputs, use_fp16, use_buffer_share)

                    io_binding.synchronize_inputs()
                    model_or_sess.run_with_iobinding(io_binding)
                    io_binding.synchronize_outputs()

                    # Sample with argmax (greedy search)
                    next_token_logits = outputs[logits_name][:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                    # Check if we previously reached EOS token id or if generated token id is EOS token id
                    has_eos = has_eos | next_tokens == eos_token_id

                    # Determine which new tokens to add to list of all token ids
                    # Add EOS token ids for batch entries that ended early (ragged batching scenario where some batch entries ended early and some haven't)
                    tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id).reshape([batch_size, 1])
                    all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

                    # Return early if all batch entries have reached EOS token id
                    if torch.all(has_eos):
                        final_fence = time.time()
                        if rank == 0:
                            print(f"Time for prompt: {1000 * (prompt_fence - start)}ms", f"Time for token: {1000 * (final_fence - prompt_fence) / count}ms")
                        break

                    # Update inputs for next inference run
                    inputs[input_ids_name] = tokens_to_add.to(torch.int64)
                    inputs[attn_mask_name] = torch.cat(
                        [inputs[attn_mask_name], (~has_eos).to(torch.int64).reshape(batch_size, 1)], 1
                    )
                    # inputs[pos_ids_name] = (
                    #     torch.max(inputs[pos_ids_name], dim=1)[0].reshape(batch_size, 1) + 1
                    # )

                    current_length += 1
                    count += 1

                    if current_length == max_length:
                        final_fence = time.time()
                        if rank == 0:
                            print(f"Time for prompt: {1000 * (prompt_fence - start)}ms", f"Time for token: {1000 * (final_fence - prompt_fence) / count}ms")
                        break

                    # Set logits to zeros for next inference run and re-use memory buffer
                    if outputs[logits_name].shape[1] != 1:
                        outputs[logits_name] = outputs[logits_name][:, :1, :].contiguous()
                    outputs[logits_name].zero_()

                # delete inputs and outputs
                keys_i = list(inputs.keys())
                keys_o = list(outputs.keys())
                for k in keys_i:
                    del inputs[k]
                for k in keys_o:
                    del outputs[k]

session_options = ort.SessionOptions()
provider_opt = {"device_id": rank, }
onnx_model_path = f"/wy/ORT_GENAI/wangye/shard/src/python/py/models/example-models/mixtral_rank_{rank}/model.onnx"
sess = ort.InferenceSession(str(onnx_model_path), providers=[(
    "CUDAExecutionProvider", provider_opt)], sess_options=session_options)

infer_model(tensor_parallel_size, rank, sess)
