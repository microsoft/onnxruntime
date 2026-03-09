# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
from argparse import ArgumentParser
from copy import deepcopy
from enum import IntEnum

from olive.cli.base import (
    BaseOliveCLICommand,
    add_input_model_options,
    add_logging_options,
    add_save_config_file_options,
    add_shared_cache_options,
    add_telemetry_options,
    get_diffusers_input_model,
    get_input_model_config,
    update_shared_cache_options,
)
from olive.common.utils import set_nested_dict_value
from olive.model.utils.diffusers_utils import is_valid_diffusers_model
from olive.telemetry import action


class ModelBuilderAccuracyLevel(IntEnum):
    fp32 = 1
    fp16 = 2
    bf16 = 3
    int8 = 4


def parse_dim_dict(s):
    try:
        return {k: int(v) if v.isdigit() else v for k, v in (item.split("=") for item in s.split(","))}
    except Exception as exc:
        raise argparse.ArgumentTypeError("Format must be key=value,... with positive integers as values") from exc


class CaptureOnnxGraphCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "capture-onnx-graph",
            help=(
                "Capture ONNX graph using PyTorch Exporter or Model Builder "
                "from the Huggingface model or PyTorch model."
            ),
        )

        # model options
        add_input_model_options(
            sub_parser,
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_diffusers=True,
            default_output_path="onnx-model",
        )

        sub_parser.add_argument(
            "--conversion_device",
            type=str,
            default="cpu",
            choices=["cpu", "gpu"],
            help="The device used to run the model to capture the ONNX graph.",
        )

        # PyTorch Exporter options
        pte_group = sub_parser.add_argument_group("PyTorch Exporter options")
        pte_group.add_argument(
            "--use_dynamo_exporter",
            action="store_true",
            help="Whether to use dynamo_export API to export ONNX model.",
        )
        pte_group.add_argument(
            "--fixed_param_dict",
            type=parse_dim_dict,
            required=False,
            help=(
                "Fix dynamic input shapes by providing a dictionary of dimension names and values, "
                "e.g., 'batch_size=1,max_length=128'"
            ),
        )
        pte_group.add_argument(
            "--past_key_value_name",
            type=str,
            default="past_key_values",
            help=(
                "The arguments name to point to past key values. For model loaded from huggingface, "
                "it is 'past_key_values'. Basically, it is used only when `use_dynamo_exporter` is True."
            ),
        )
        pte_group.add_argument(
            "--torch_dtype",
            type=str,
            help=(
                "The dtype to cast the model to before capturing the ONNX graph, e.g., 'float32' or 'float16'."
                " If not specified will use the model as is."
            ),
        )
        pte_group.add_argument(
            "--target_opset",
            type=int,
            default=20,
            help="The target opset version for the ONNX model. Default is 20.",
        )

        # Model Builder options
        mb_group = sub_parser.add_argument_group("Model Builder options")
        mb_group.add_argument(
            "--use_model_builder",
            action="store_true",
            help="Whether to use Model Builder to capture ONNX model.",
        )
        mb_group.add_argument(
            "--precision",
            type=str,
            default="fp16",
            choices=["fp16", "fp32", "int4", "bf16"],
            help="The precision of the ONNX model. This is used by Model Builder",
        )
        mb_group.add_argument(
            "--int4_block_size",
            type=int,
            required=False,
            choices=[16, 32, 64, 128, 256],
            help="Specify the block_size for int4 quantization. Acceptable values: 16/32/64/128/256.",
        )
        mb_group.add_argument(
            "--int4_accuracy_level",
            type=ModelBuilderAccuracyLevel,
            required=False,
            help="Specify the minimum accuracy level for activation of MatMul in int4 quantization.",
        )
        mb_group.add_argument(
            "--exclude_embeds",
            type=bool,
            default=False,
            required=False,
            help="Remove embedding layer from your ONNX model.",
        )
        mb_group.add_argument(
            "--exclude_lm_head",
            type=bool,
            default=False,
            required=False,
            help="Remove language modeling head from your ONNX model.",
        )
        mb_group.add_argument(
            "--enable_cuda_graph",
            type=bool,
            default=None,  # Explicitly setting to None to differentiate between user intent and default.
            required=False,
            help=(
                "The model can use CUDA graph capture for CUDA execution provider. "
                "If enabled, all nodes being placed on the CUDA EP is the prerequisite "
                "for the CUDA graph to be used correctly."
            ),
        )
        mb_group.add_argument(
            "--extra_mb_options",
            type=str,
            required=False,
            help="Extra key-value pairs options to pass to the model builder. e.g., 'int4_is_symmetric=true,int4_op_types_to_quantize=MatMul/Gemm'.",
        )

        sub_parser.add_argument(
            "--use_ort_genai", action="store_true", help="Use OnnxRuntime generate() API to run the model"
        )

        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_shared_cache_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=CaptureOnnxGraphCommand)

    @action
    def run(self):
        return self._run_workflow()

    def _get_run_config(self, tempdir: str) -> dict:
        config = deepcopy(TEMPLATE)

        # Check if diffusers model detection is needed
        is_diffusers = is_valid_diffusers_model(self.args.model_name_or_path) if self.args.model_name_or_path else False
        if is_diffusers:
            input_model_config = get_diffusers_input_model(self.args, self.args.model_name_or_path)
        else:
            input_model_config = get_input_model_config(self.args)
        assert input_model_config["type"].lower() in {
            "hfmodel",
            "pytorchmodel",
            "diffusersmodel",
        }, "Only HfModel, PyTorchModel, and DiffusersModel are supported in capture-onnx-graph command."

        is_diffusers_model = input_model_config["type"].lower() == "diffusersmodel"

        # whether model is in fp16 or bf16 (currently not supported by CPU EP)
        is_fp16_or_bf16 = (not self.args.use_model_builder and self.args.torch_dtype == "float16") or (
            self.args.use_model_builder and self.args.precision in ("fp16", "bf16")
        )
        to_replace = [
            ("input_model", input_model_config),
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
            (("systems", "local_system", "accelerators", 0, "device"), "gpu" if is_fp16_or_bf16 else "cpu"),
            (
                ("systems", "local_system", "accelerators", 0, "execution_providers"),
                [("CUDAExecutionProvider" if is_fp16_or_bf16 else "CPUExecutionProvider")],
            ),
        ]

        if is_diffusers_model:
            del config["passes"]["m"]
            to_replace.extend(
                [
                    (
                        ("passes", "c", "device"),
                        self.args.conversion_device if self.args.conversion_device == "cpu" else "cuda",
                    ),
                    (("passes", "c", "torch_dtype"), self.args.torch_dtype),
                    (("passes", "c", "target_opset"), self.args.target_opset),
                ]
            )
        elif self.args.use_model_builder:
            del config["passes"]["c"]
            to_replace.extend(
                [
                    (("passes", "m", "precision"), self.args.precision),
                    (("passes", "m", "exclude_embeds"), self.args.exclude_embeds),
                    (("passes", "m", "exclude_lm_head"), self.args.exclude_lm_head),
                    (("passes", "m", "enable_cuda_graph"), self.args.enable_cuda_graph),
                ]
            )
            if self.args.extra_mb_options:
                to_replace.append(
                    (
                        ("passes", "m", "extra_options"),
                        BaseOliveCLICommand._parse_extra_options(self.args.extra_mb_options.split(",")),
                    )
                )
            if self.args.int4_block_size is not None:
                to_replace.append((("passes", "m", "int4_block_size"), self.args.int4_block_size))
            if self.args.int4_accuracy_level is not None:
                to_replace.append((("passes", "m", "int4_accuracy_level"), self.args.int4_accuracy_level))
        else:
            to_replace.extend(
                [
                    (
                        ("passes", "c", "device"),
                        self.args.conversion_device if self.args.conversion_device == "cpu" else "cuda",
                    ),
                    (("passes", "c", "torch_dtype"), self.args.torch_dtype),
                    (("passes", "c", "target_opset"), self.args.target_opset),
                    (("passes", "c", "use_dynamo_exporter"), self.args.use_dynamo_exporter),
                    (("passes", "c", "save_metadata_for_token_generation"), self.args.use_ort_genai),
                ]
            )
            if self.args.use_dynamo_exporter:
                to_replace.append((("passes", "c", "past_key_value_name"), self.args.past_key_value_name))
            if not self.args.use_ort_genai:
                del config["passes"]["m"]
            else:
                mb_precision = {
                    "fp16": "fp16",
                    "bf16": "bf16",
                }.get(self.args.precision, "fp32")

                to_replace.extend(
                    [
                        (("passes", "m", "precision"), mb_precision),
                        (("passes", "m", "metadata_only"), True),
                    ]
                )
        if self.args.fixed_param_dict:
            to_replace.append((("passes", "f", "dim_param"), list(self.args.fixed_param_dict.keys())))
            to_replace.append((("passes", "f", "dim_value"), list(self.args.fixed_param_dict.values())))
        else:
            del config["passes"]["f"]
        for keys, value in to_replace:
            if value is None:
                continue
            set_nested_dict_value(config, keys, value)
        update_shared_cache_options(config, self.args)

        return config


TEMPLATE = {
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            # might need an ep option to set for model builder, it is sensitive to ep
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "passes": {
        "c": {
            "type": "OnnxConversion",
        },
        "m": {"type": "ModelBuilder", "metadata_only": False},
        "f": {"type": "DynamicToFixedShape"},
    },
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}
