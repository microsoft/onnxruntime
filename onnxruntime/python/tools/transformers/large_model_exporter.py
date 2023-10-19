# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Export LLM to onnx
"""
import argparse
import inspect
import math
import os
import re
import tempfile
from pathlib import Path

import onnx
import torch
import transformers
from torch import nn


def disable_huggingface_init():
    """do not init model twice as it slow initialization"""

    torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.constant_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.orthogonal_ = lambda x, *args, **kwargs: x


def get_model_parameter_size(model: nn.Module):
    """to calculate how much memory this model needs"""
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    return all_size


def initialize_model_and_sample_inputs(hf_model: str, tokenizer=None):
    """
    get the pretrained torch model from hugginface,
    and onnx_model_name, sample model-inputs
    """
    onnx_model_name = Path(hf_model + "/").name

    onnx_model_name = re.sub(r"[^0-9a-zA-Z]", onnx_model_name, "_") + ".onnx"
    disable_huggingface_init()

    model = transformers.AutoModelForCausalLM.from_pretrained(  # type: ignore
        hf_model, torch_dtype=torch.float16, trust_remote_code=True
    )
    if tokenizer is None:
        tokenizer = hf_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)  # type: ignore

    sample_inputs = tuple(tokenizer("Hello, my dog is cute", return_tensors="pt").values())
    return onnx_model_name, model, sample_inputs


def auto_pipeline_parallel(model: nn.Module, gpulist: list, sample_inputs: tuple):
    """Make the model executable across multiple GPUs."""

    def input_gpu_device_hook(mod, inputs, kwargs):
        modifyed_inputs = []
        first_dev = None
        for layer_input in inputs:
            if type(layer_input) is not torch.Tensor:
                modifyed_inputs.append(layer_input)
            elif hasattr(mod, "weight"):
                modifyed_inputs.append(layer_input.to(mod.weight.device))
            elif hasattr(mod, "parameters"):
                device = next(mod.parameters(), layer_input).device
                modifyed_inputs.append(layer_input.to(device))
            elif hasattr(next(mod.children(), None), "weight"):
                modifyed_inputs.append(layer_input.to(next(mod.children()).weight.device))
            elif first_dev is not None and layer_input.device != first_dev:
                modifyed_inputs.append(layer_input.to(first_dev))
            else:
                modifyed_inputs.append(layer_input)
            if first_dev is None:
                first_dev = modifyed_inputs[0].device
        for key, value in kwargs.items():
            if type(value) is torch.Tensor:
                kwargs[key] = value.to(first_dev)

        return (tuple(modifyed_inputs), kwargs)

    def move_layer_to_device_rurc(mod, dev):
        mod.to(dev)
        for layer in mod.named_children():
            move_layer_to_device_rurc(layer[1], dev)

    model = model.half()
    all_hooks = []
    # model.register_module_forward_pre_hook(input_gpu_device_hook)
    all_hooks.append(model.register_forward_pre_hook(input_gpu_device_hook, with_kwargs=True))
    pre_fix = next(iter(model.named_children()))[0]
    for top_name, top_module in model.named_children():
        for name, module in top_module.named_children():
            all_hooks.append(module.register_forward_pre_hook(input_gpu_device_hook, with_kwargs=True))
            if type(module) in [torch.nn.ModuleList]:
                num_layers_on_each_gpu = math.floor(len(module) / len(gpulist))
                for idx, attn_layer in enumerate(module):
                    all_hooks.append(attn_layer.register_forward_pre_hook(input_gpu_device_hook, with_kwargs=True))

                    to_dev = gpulist[min(idx // num_layers_on_each_gpu, len(gpulist))]
                    attn_layer.to(to_dev)
                    move_layer_to_device_rurc(attn_layer, to_dev)
                    print(f"move {pre_fix}.{name}.{idx} to {to_dev}")
            else:
                module.to(gpulist[0])
                print(f"move {pre_fix}.{name} to {gpulist[0]}")
        if len(list(top_module.named_children())) == 0:
            top_module.to(gpulist[0])
            print(f"move {top_name} to {gpulist[0]}")

    with torch.no_grad():
        model(sample_inputs[0], attention_mask=sample_inputs[1])
    return model


def retrieve_onnx_inputs(model, sample_inputs):
    """
    auto retrieve onnx inputs from torch model as we can't enumlate all possibilities
    for all models
    """
    user_inputs = []

    def hook_for_inputs(_, inputs, kwargs):
        user_inputs.append((inputs, kwargs))
        return user_inputs[0]

    hook_handle = model.register_forward_pre_hook(hook_for_inputs, with_kwargs=True)

    forward_params = inspect.signature(model.forward).parameters
    input_keys = list(forward_params.keys())
    default_values = [forward_params.get(key).default for key in input_keys]
    model(sample_inputs[0], attention_mask=sample_inputs[1])
    hook_handle.remove()
    user_inputs = user_inputs[0]
    onnx_inputs = default_values
    for idx, _val in enumerate(user_inputs[0]):
        onnx_inputs[idx] = user_inputs[0][idx]
    for key, value in user_inputs[1].items():
        idx = input_keys.index(key)
        onnx_inputs[idx] = value
    for idx, (key, value) in enumerate(zip(input_keys, onnx_inputs)):
        if type(value) is torch.Tensor:
            value.to(model.device)
        # Didn't touch past_key_value now, please change it if you want
        if "use_cache" in key:
            onnx_inputs[idx] = False

    return input_keys, tuple(onnx_inputs)


def move_to_approprate_device(model: nn.Module, sample_inputs_tp: tuple) -> nn.Module:
    """
    According to the model size, we will upload it to
    CPU if has no GPU or enough GPU memory,
    Single GPU if has only one GPU in local or model size is enough to fit one GPU
    Multiple GPU if there is more than one gpu in local and model is too large
    """
    total_mem_per_cpu = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    print(f"Model_Size = {get_model_parameter_size(model)/1024} GB")
    print(f"total_mem_per_cpu = {total_mem_per_cpu/1024} GB")
    if get_model_parameter_size(model) > total_mem_per_cpu * 0.45:
        device_collection = [torch.device(i) for i in range(torch.cuda.device_count())]
        if len(device_collection) > 1:
            print(
                f"{len(device_collection)} GPUs are used to export onnx, \
                   Please set CUDA_VISIBLE_DEVICES to use specific GPU group"
            )
            model = auto_pipeline_parallel(model, device_collection, sample_inputs_tp)
        else:
            print("!!!! convert model to float and export onnx using CPU")
            model = model.cpu().float()
    else:
        print("Export model on a single GPU")
        model = model.cuda().half()
    return model


def adapt_inputs_to_device(sample_inputs: tuple, device: torch.Device) -> tuple:
    """move inputs to device"""
    sample_inputs_ = []
    for sample_int in sample_inputs:
        if isinstance(sample_int, torch.Tensor):
            sample_inputs_.append(sample_int.to(device))
        else:
            sample_inputs_.append(sample_int)
    return tuple(sample_inputs_)


@torch.no_grad()
def export_onnx(hf_model: str, onnx_path_str: str, opset: int = 17):
    """
    do export
    model: torch model
    onnx_path: where the onnx model saved to
    sample_inputs_tp: inputs for torch model
    """
    onnx_model_name, model, sample_inputs_tp = initialize_model_and_sample_inputs(hf_model)

    model = move_to_approprate_device(model, sample_inputs_tp)

    sample_inputs = adapt_inputs_to_device(sample_inputs_tp, next(model.parameters()).device)

    # input_keys would be usesful if the model has some special inputs
    input_keys, onnx_inputs = retrieve_onnx_inputs(model, sample_inputs)

    onnx_path: Path = Path(onnx_path_str).absolute()
    if onnx_path.suffix != ".onnx":
        onnx_path = onnx_path / onnx_model_name

    # two step to export onnx
    # 1. export onnx with lots of pieces of weights
    # 2. save all weights to external data
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_onnx = os.path.join(tmpdirname, "tmp.onnx")

        onnx_inp_names = ("input_ids", "attention_mask")
        onnx_out_names = ("logits",)
        onnx_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
        }
        torch.onnx.export(
            model=model,
            args=onnx_inputs,
            f=tmp_onnx,
            verbose=False,
            opset_version=opset,
            input_names=onnx_inp_names,
            output_names=onnx_out_names,
            dynamic_axes=onnx_dynamic_axes,
        )

        onnx_path.unlink(missing_ok=True)
        (onnx_path.parent / f"{onnx_model_name}_ext.data").unlink(missing_ok=True)

        onnx_model = onnx.load(str(tmp_onnx))
        onnx.save_model(
            onnx_model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{onnx_model_name}_ext.data",
            size_threshold=1024,
            convert_attribute=False,
        )


def parse_arguments():
    """
    arguments parsing.
    model:
    onnx_path:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        default=["meta-llama/Llama-2-70b-hf"],
        help="Pre-trained models in huggingface model hub",
    )
    parser.add_argument(
        "-s",
        "--saved_path",
        required=False,
        type=str,
        default="./onnx_models/",
        help="where the onnx model will be saved",
    )
    parser.add_argument(
        "--opset",
        required=False,
        type=int,
        default=17,
        help=(
            "the opset to save onnx model, \
              try to increase it if this opset doens't have new features you want"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    export_onnx(args.model, args.saved_path, args.opset)
