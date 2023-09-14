import argparse
import logging
import os
import sys
import tempfile
from itertools import chain
from typing import List

import onnx
import torch

from transformers import LlamaConfig, LlamaForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from onnxruntime.transformers.benchmark_helper import Precision, prepare_environment, setup_logger  # noqa: E402

# from onnxruntime.transformers.onnx_model import OnnxModel  # noqa: E402

logger = logging.getLogger("")


def get_model_inputs(config: LlamaConfig, device):
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.int64, device=device
    )
    attn_mask = torch.randint(low=0, high=2, size=(batch_size, seq_len), dtype=torch.int64, device=device)
    # pos_ids is of shape (batch_size, seq_len)
    pos_ids = attn_mask.long().cumsum(-1) - 1
    pos_ids.masked_fill_(attn_mask == 0, 1)

    return (input_ids, attn_mask, pos_ids)


def get_model_with_past_kv_inputs(config: LlamaConfig, num_kv_heads, head_dim, device):
    batch_size, past_seq_len = 2, 8
    num_heads, head_size = num_kv_heads, head_dim
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, 1), dtype=torch.int64, device=device)
    attn_mask = torch.randint(low=0, high=2, size=(batch_size, past_seq_len + 1), dtype=torch.int64, device=device)
    # pos_ids is of shape (batch_size, 1)
    pos_ids = attn_mask.long().cumsum(-1) - 1
    pos_ids.masked_fill_(attn_mask == 0, 1)
    pos_ids = pos_ids[:, -1].unsqueeze(-1)
    past_kv = [
        (
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=config.torch_dtype),
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=config.torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]

    return (input_ids, attn_mask, pos_ids, past_kv)


def get_model_dynamic_axes(input_names: List[str], output_names: List[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in input_names:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + 1, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length + 1"}
        else:
            raise Exception("Unknown input or output name found")
    return dynamic_axes


def get_model_with_past_kv_dynamic_axes(input_names: List[str], output_names: List[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"input_ids", "position_ids"}:
            # shape is (batch_size, 1)
            dynamic_axes[name] = {0: "batch_size"}
        elif name == "attention_mask":
            # shape is (batch_size, past_sequence_length + 1)
            dynamic_axes[name] = {0: "batch_size", 1: "past_sequence_length + 1"}
        elif "past" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        elif name == "logits":
            # shape is (batch_size, 1, vocab_size)
            dynamic_axes[name] = {0: "batch_size"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + 1, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length + 1"}
        else:
            raise Exception("Unknown input or output name found")
    return dynamic_axes


def save_onnx_model(onnx_model: onnx.ModelProto, output_path: str, data_path: str):
    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
        size_threshold=1024,
        convert_attribute=False,
    )


# Notes:
# 1) Dynamo export will not work automatically until this issue is resolved: https://github.com/microsoft/onnxscript/issues/493
#
# 2) Dynamo export will run manually if you set the ONNX file path to the same path that you use to save the model after export.
# In other words, the value of `temp_path` should be set as the ONNX file path. You can open the issue in your browser to find
# the location in ONNX Script where you have to make this change.
#
# Once the issue is resolved, we can modify the code below as follows for each export.
#
# Before:
# temp_dir = args.output_name
# temp_path = os.path.join(temp_dir, "temp.onnx")
# ...
# ...
# ...
# del onnx_model
# os.system(f"rm {os.path.join(temp_dir, 'model.*')} && rm {os.path.join(temp_dir, '*.weight')} && rm {temp_path}")
#
#
# After:
# temp_dir = tempfile.TemporaryDirectory()
# temp_path = os.path.join(temp_dir.name, "temp.onnx")
# ...
# ...
# ...
# del onnx_model
# temp_dir.cleanup()
#
#
# 3) Once exported, you will need to manually remove the external data that is saved to individual files. Do NOT delete the
# file with the ".onnx.data" extension.
def run_dynamo_export(args: argparse.Namespace, l_config: LlamaConfig, llama: LlamaForCausalLM, rank=0, world_size=1):
    from torch._dynamo import config

    config.capture_scalar_outputs = True

    # Export decoder_model.onnx
    input_ids, attn_mask, pos_ids = get_model_inputs(l_config, llama.device)
    temp_dir = args.output_name  # tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir, "temp.onnx")  # os.path.join(temp_dir.name, "temp.onnx")
    torch.onnx.dynamo_export(
        llama, input_ids, attn_mask, pos_ids, export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
    ).save(temp_path)

    # Check decoder_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    output_path = f"{args.model_name}_decoder_model_fp32.onnx"
    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(onnx_model, output_path, f"{args.model_name}_decoder_model_fp32.onnx.data")
    del onnx_model
    os.system(
        f"rm {os.path.join(temp_dir, 'model.*')} && rm {os.path.join(temp_dir, '*.weight')} && rm {temp_path}"
    )  # temp_dir.cleanup()

    # Export decoder_with_past_model.onnx
    head_dim = l_config.hidden_size // l_config.num_attention_heads
    num_kv_heads = l_config.num_key_value_heads // world_size
    input_ids, attn_mask, pos_ids, past_kv = get_model_with_past_kv_inputs(
        l_config, num_kv_heads, head_dim, llama.device
    )
    temp_dir = args.output_name  # tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir, "temp.onnx")  # os.path.join(temp_dir.name, "temp.onnx")
    torch.onnx.dynamo_export(
        llama, input_ids, attn_mask, pos_ids, past_kv, export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
    ).save(temp_path)

    # Check decoder_with_past_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    output_path = f"{args.model_name}_decoder_with_past_model_fp32.onnx"
    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(onnx_model, output_path, f"{args.model_name}_decoder_with_past_model_fp32.onnx.data")
    del onnx_model
    os.system(
        f"rm {os.path.join(temp_dir, 'model.*')} && rm {os.path.join(temp_dir, '*.weight')} && rm {temp_path}"
    )  # temp_dir.cleanup()

    logger.info(f"The {args.model_name} ONNX model has been successfully created with the Dynamo exporter!")


OUTPUT_PATH = "rank-{}_decoder_model_fp32.onnx"
OUTPUT_WITH_PAST_PATH = "rank-{}_decoder_with_past_model_fp32.onnx"


def run_torchscript_export(
    args: argparse.Namespace, l_config: LlamaConfig, llama: LlamaForCausalLM, rank=0, world_size=1
):
    # Export decoder_model.onnx
    decoder_inputs = get_model_inputs(l_config, llama.device)

    output_path = OUTPUT_PATH.format(rank)
    output_with_past_path = OUTPUT_WITH_PAST_PATH.format(rank)
    if os.path.exists(output_path):
        return output_path, output_with_past_path

    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = [
        "logits",
        *list(
            chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(l_config.num_hidden_layers))
        ),
    ]
    dynamic_axes = get_model_dynamic_axes(input_names, output_names)
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, "temp.onnx")
    torch.onnx.export(
        llama,
        args=decoder_inputs,
        f=temp_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        # custom_opsets={"com.microsoft": 1},
        verbose=args.verbose,
    )

    # Check decoder_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(onnx_model, output_path, f"{output_path}.data")
    del onnx_model
    temp_dir.cleanup()

    # Export decoder_with_past_model.onnx
    head_dim = l_config.hidden_size // l_config.num_attention_heads
    num_kv_heads = l_config.num_key_value_heads // world_size
    decoder_with_past_inputs = get_model_with_past_kv_inputs(l_config, num_kv_heads, head_dim, llama.device)
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *list(
            chain.from_iterable(
                (f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in range(l_config.num_hidden_layers)
            )
        ),
    ]
    output_names = [
        "logits",
        *list(
            chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(l_config.num_hidden_layers))
        ),
    ]
    dynamic_axes = get_model_with_past_kv_dynamic_axes(input_names, output_names)
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, "temp.onnx")
    torch.onnx.export(
        llama,
        args=decoder_with_past_inputs,
        f=temp_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        verbose=args.verbose,
    )

    # Check decoder_with_past_model.onnx and save all external data to one file
    onnx.checker.check_model(temp_path)
    onnx.shape_inference.infer_shapes_path(temp_path)

    onnx_model = onnx.load_model(temp_path, load_external_data=True)
    save_onnx_model(onnx_model, output_with_past_path, f"{output_with_past_path}.data")
    del onnx_model
    temp_dir.cleanup()

    logger.info(f"The {args.model} ONNX model has been successfully created with the TorchScript exporter!")

    return output_path, output_with_past_path


def optimize_onnx_model(rank=0):
    from onnxruntime.transformers.optimizer import optimize_model

    output_path = OUTPUT_PATH.format(rank)
    output_with_past_path = OUTPUT_WITH_PAST_PATH.format(rank)

    opt_model_path = f"opt_{output_path}"
    opt_model_with_past_path = f"opt_{output_with_past_path}"

    if os.path.exists(opt_model_path) and os.path.exists(opt_model_with_past_path):
        return opt_model_path, opt_model_with_past_path

    output_model = optimize_model(
        output_path,
        model_type="t5",
        opt_level=0,
        use_gpu=False,
        only_onnxruntime=False,
    )

    output_model.save_model_to_file(opt_model_path, True, all_tensors_to_one_file=True)

    del output_model

    output_with_past_model = optimize_model(
        output_with_past_path,
        model_type="t5",
        opt_level=0,
        use_gpu=False,
        only_onnxruntime=False,
    )

    output_with_past_model.save_model_to_file(opt_model_with_past_path, True, all_tensors_to_one_file=True)

    return opt_model_path, opt_model_with_past_path
