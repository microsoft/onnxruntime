"""Inference-only Mixtral model."""
from multiprocessing import Process, set_start_method
from vllm.worker.worker import _init_distributed_environment
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tensor_model_parallel_group)
from vllm.config import ParallelConfig
from torch import nn
import time
import numpy as np
import os
import torch
from pathlib import Path
import onnxruntime as ort
from vllm import paged_attn
torch.zeros(1).cuda()

torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

input_ids_name = "input_ids"
sequence_length_name = "seqlens_k"

#-----------------------settings-----------------------
use_int4 = True

tensor_parallel_size = 1 if use_int4 else 4
logits_name = "last_hidden_state"
past_key_name = "present_key."
past_value_name = "present_values."
present_key_name = "past_key."
present_value_name = "past_values."

o_cache_name_2 = [
'631',
'632',
'859',
'860',
'1087',
'1088',
'1315',
'1316',
'1543',
'1544',
'1771',
'1772',
'1999',
'2000',
'2227',
'2228',
'2455',
'2456',
'2683',
'2684',
'2911',
'2912',
'3139',
'3140',
'3367',
'3368',
'3595',
'3596',
'3823',
'3824',
'4051',
'4052',
'4279',
'4280',
'4507',
'4508',
'4735',
'4736',
'4963',
'4964',
'5191',
'5192',
'5419',
'5420',
'5647',
'5648',
'5875',
'5876',
'6103',
'6104',
'6331',
'6332',
'6559',
'6560',
'6787',
'6788',
'7015',
'7016',
'7243',
'7244',
'7471',
'7472',
'7699',
'7700',
]

o_cache_name = [
'629',
'630',
'857',
'858',
'1085',
'1086',
'1313',
'1314',
'1541',
'1542',
'1769',
'1770',
'1997',
'1998',
'2225',
'2226',
'2453',
'2454',
'2681',
'2682',
'2909',
'2910',
'3137',
'3138',
'3365',
'3366',
'3593',
'3594',
'3821',
'3822',
'4049',
'4050',
'4277',
'4278',
'4505',
'4506',
'4733',
'4734',
'4961',
'4962',
'5189',
'5190',
'5417',
'5418',
'5645',
'5646',
'5873',
'5874',
'6101',
'6102',
'6329',
'6330',
'6557',
'6558',
'6785',
'6786',
'7013',
'7014',
'7241',
'7242',
'7469',
'7470',
'7697',
'7698',
]

o_name_dict = {}
for i in range(32):
    o_name_dict[present_key_name + f"{i}"] = o_cache_name[i*2 + 0]
    o_name_dict[present_value_name + f"{i}"] = o_cache_name[i*2 + 1]

o_name_dict_2 = {}
for i in range(32):
    o_name_dict_2[present_key_name + f"{i}"] = o_cache_name_2[i*2 + 0]
    o_name_dict_2[present_value_name + f"{i}"] = o_cache_name_2[i*2 + 1]

pt_to_np = {
    "torch.int32": np.int32,
    "torch.int64": np.int64,
    "torch.float32": np.float32,
    "torch.float16": np.float16
}

def init_test_distributed_environment(tensor_parallel_size: int, rank: int,
                                      distributed_init_port: str = "51408"):
    parallel_config = ParallelConfig(1, tensor_parallel_size,
                                     worker_use_ray=True)
    distributed_init_method = f"tcp://127.0.0.1:{distributed_init_port}"
    torch.cuda.set_device(rank)
    _init_distributed_environment(
        parallel_config, rank, distributed_init_method)

def get_initial_inputs_and_outputs_for_bench(batch_size, sequence_length, tensor_parallel_size, device: torch.device, use_fp16: bool, use_buffer_share: bool, rank):
    name_dict = o_name_dict if rank == 0 else o_name_dict_2
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    input_ids = torch.randint(0, 32000, (batch_size, sequence_length), device=device, dtype=torch.int64)
    sequence_length = torch.ones(batch_size, device=device, dtype=torch.int64) * sequence_length

    inputs = {
        input_ids_name: input_ids.contiguous(),
        sequence_length_name: sequence_length.contiguous(),
    }

    batch_size, sequence_length = input_ids.shape
    max_sequence_length = 1024 # subject to change
    num_heads, head_size = int(8 / tensor_parallel_size), 128
    for i in range(32):
        past_key = torch.zeros(batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, device=device, dtype=torch_dtype)
        past_value = torch.zeros(batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, device=device, dtype=torch_dtype)
        inputs.update({
            past_key_name + f"{i}": past_key.contiguous(),
            past_value_name + f"{i}": past_value.contiguous(),
        })

    logits = torch.zeros(batch_size, sequence_length, 32000, device=device, dtype=torch_dtype)
    outputs = {
        logits_name: logits.contiguous()
    }
    if not use_buffer_share:
        for i in range(32):
            present_key = torch.zeros(batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype)
            present_value = torch.zeros(batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype)
            outputs.update({
                name_dict[present_key_name + f"{i}"]: present_key.contiguous(),
                name_dict[present_value_name + f"{i}"]: present_value.contiguous(),
            })

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
            v = inputs[name.replace("present", "past")]
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
    name_dict = o_name_dict if rank == 0 else o_name_dict_2
    device = torch.device(rank)
    use_fp16 = True
    use_buffer_share = False # buffer sharing is not supported now
    eos_token_id = 50256 # never let it reach eos for benchmark
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    for token_num in [32]:
        for batch_size in [2]:
            for seq_len in [512]:
                max_length = seq_len + token_num
                if rank == 0:
                    print("batch size:", batch_size, "seq len:", seq_len, "token_num:", token_num)
                # Get model and its initial inputs/outputs
                inputs, outputs = get_initial_inputs_and_outputs_for_bench(batch_size, seq_len, tensor_parallel_size, device, use_fp16, use_buffer_share, rank)

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
                    inputs[sequence_length_name] = torch.ones(batch_size, device=device, dtype=torch.int64) * (current_length)
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

                    if not use_buffer_share:
                        for i in range(32):
                            inputs[past_key_name + f"{i}"] = outputs[name_dict[present_key_name + f"{i}"]]
                            inputs[past_value_name + f"{i}"] = outputs[name_dict[present_value_name + f"{i}"]]

                        new_sequence_length = current_length
                        local_num_heads, head_size = int(8 / tensor_parallel_size), 128
                        for i in range(32):
                            present_key = torch.zeros(batch_size, local_num_heads, new_sequence_length, head_size, device=device, dtype=torch_dtype)
                            present_value = torch.zeros(batch_size, local_num_heads, new_sequence_length, head_size, device=device, dtype=torch_dtype)
                            outputs.update({
                               name_dict[present_key_name + f"{i}"]: present_key.contiguous(),
                               name_dict[present_value_name + f"{i}"]: present_value.contiguous(),
                            })

def test_model_load(tensor_parallel_size, rank):
    init_test_distributed_environment(tensor_parallel_size, rank)

    os.environ['LOCAL_WORLD_SIZE'] = str(tensor_parallel_size)
    os.environ['LOCAL_RANK'] = str(rank)
    torch.cuda.set_device(rank)

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(paged_attn.__file__)
    provider_opt = {"device_id": rank, }

    onnx_model_path = Path(f"./mixtral_fp16_tp4/mixtral_with_past_rank{rank}.onnx").absolute() if not use_int4 else Path(f"/home/jicwen/work/vllm/mixtral_infer/onnx_models_int4/mixtral_with_past_rank{rank}_int4.onnx").absolute()
    #onnx_model_path = Path(f"/home/jicwen/work/vllm/mixtral_infer/onnx_models/mixtral_with_past_rank{rank}.onnx").absolute()
    sess = ort.InferenceSession(str(onnx_model_path), providers=[(
        "CUDAExecutionProvider", provider_opt)], sess_options=session_options)

    infer_model(tensor_parallel_size, rank, sess)


def process_entry(tensor_parallel_size, rank):
    test_model_load(tensor_parallel_size, rank)


if __name__ == "__main__":
    if tensor_parallel_size == 1:
        test_model_load(tensor_parallel_size, 0)
    else:
        set_start_method("spawn", force=True)

        processes = []
        for rank in range(tensor_parallel_size):
            p = Process(target=process_entry,
                        args=(tensor_parallel_size, rank))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
