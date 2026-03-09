# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser

from huggingface_hub.constants import HF_HUB_CACHE

from olive.cli.base import BaseOliveCLICommand, add_logging_options, add_telemetry_options
from olive.common.utils import WeightsFileFormat, save_weights
from olive.telemetry import action


class ExtractAdaptersCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "extract-adapters",
            help="Extract LoRAs from PyTorch model to separate files",
        )
        sub_parser.add_argument(
            "-m",
            "--model_name_or_path",
            type=str,
            required=True,
            help="Path to the PyTorch model. Can be a local folder or Hugging Face id.",
        )
        sub_parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=[el.value for el in WeightsFileFormat],
            required=True,
            help="Format to save the LoRAs in.",
        )
        sub_parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            default="adapters",
            help="Output folder to save the LoRAs in the requested format.",
        )
        sub_parser.add_argument(
            "--dtype",
            type=str,
            default="float32",
            choices=["float32", "float16"],
            help="Data type to save LoRAs as. Default is float32.",
        )
        sub_parser.add_argument(
            "--cache_dir",
            type=str,
            default=HF_HUB_CACHE,
            help="Cache dir to store temporary files in. Default is Hugging Face's default cache dir.",
        )
        add_logging_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=ExtractAdaptersCommand)

    @action
    def run(self):
        # Reference: https://huggingface.co/microsoft/Phi-4-multimodal-instruct-onnx/blob/05f620b467891affcb00b464e5a73e7cf2de61f9/onnx/builder.py#L318
        import os
        from pathlib import Path

        from huggingface_hub import HfApi
        from peft import AutoPeftModelForCausalLM
        from torch import float16, float32
        from transformers import AutoConfig, AutoModelForCausalLM

        torch_dtype = float16 if self.args.dtype == "float16" else float32

        # Check if model is Transformers or Peft
        is_peft = False
        adapter_paths = []
        if os.path.exists(self.args.model_name_or_path):
            for root, _, files in os.walk(self.args.model_name_or_path):
                for f in files:
                    path = os.path.join(root, f)
                    if "adapter_config.json" in path and self.args.cache_dir not in path:
                        adapter_paths.append(root)
                        is_peft = True
        else:
            is_peft = "adapter_config.json" in HfApi().list_repo_files(self.args.model_name_or_path)

        # Load LoRA config and LoRA model
        config = AutoConfig.from_pretrained(
            self.args.model_name_or_path, cache_dir=self.args.cache_dir, trust_remote_code=True
        )
        if is_peft:
            # Peft model
            first_adapter_name = Path(adapter_paths[0]).name
            peft_model = AutoPeftModelForCausalLM.from_pretrained(
                os.path.join(self.args.model_name_or_path, first_adapter_name),
                adapter_name=first_adapter_name,
                cache_dir=self.args.cache_dir,
                trust_remote_code=True,
            )
            for adapter_path in adapter_paths[1:]:
                adapter_name = Path(adapter_path).name
                peft_model.load_adapter(adapter_path, adapter_name)
        else:
            # Transformers model
            peft_model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path, cache_dir=self.args.cache_dir, trust_remote_code=True
            )

        head_size = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        query_size = config.num_attention_heads * head_size
        key_value_size = config.num_key_value_heads * head_size
        intermediate_size = config.intermediate_size

        adapter_sets = {}
        for key, val in peft_model.state_dict().items():
            # Map name in graph as key
            new_dict = {}
            key_name = (
                key.replace("self_attn", "attn").replace("lora_A", "lora_A.MatMul").replace("lora_B", "lora_B.MatMul")
            )

            if "lora_A" in key_name:
                # LoRA_A is shared across projections
                if "qkv_proj" in key_name:
                    new_dict[key_name.replace("qkv_proj", "q_proj")] = val
                    new_dict[key_name.replace("qkv_proj", "k_proj")] = val
                    new_dict[key_name.replace("qkv_proj", "v_proj")] = val
                elif "gate_up_proj" in key_name:
                    new_dict[key_name.replace("gate_up_proj", "gate_proj")] = val
                    new_dict[key_name.replace("gate_up_proj", "up_proj")] = val
                else:
                    new_dict[key_name] = val

            elif "lora_B" in key_name:
                # LoRA_B is split across projections
                if "qkv_proj" in key_name:
                    new_dict[key_name.replace("qkv_proj", "q_proj")] = val[:query_size, :]
                    new_dict[key_name.replace("qkv_proj", "k_proj")] = val[query_size : query_size + key_value_size, :]
                    new_dict[key_name.replace("qkv_proj", "v_proj")] = val[query_size + key_value_size :, :]
                elif "gate_up_proj" in key_name:
                    new_dict[key_name.replace("gate_up_proj", "gate_proj")] = val[:intermediate_size, :]
                    new_dict[key_name.replace("gate_up_proj", "up_proj")] = val[intermediate_size:, :]
                else:
                    new_dict[key_name] = val

            else:
                continue

            # Use negative indices to access `module_path` since prefix can be multiple options
            # (e.g. `model`, `model.model`, etc)
            module_path = key.split(".")
            layer_id = module_path[-6]
            class_name = module_path[-5]  # e.g. self_attn, mlp, etc.
            class_attr_name = module_path[-4]  # e.g. qkv_proj, gate_up_proj, etc.
            lora_name = module_path[-3]  # e.g. lora_A, lora_B, etc.
            adapter_name = module_path[-2]  # e.g. default, speech, etc.

            if adapter_name not in adapter_sets:
                adapter_sets[adapter_name] = {}

            prefix = "base_model.model.model" if is_peft else "model"
            scale_val = eval(  # pylint: disable=eval-used
                f"peft_model.{prefix}.layers[{layer_id}].{class_name}.{class_attr_name}.scaling['{adapter_name}']"
            )
            for new_key, new_val in new_dict.items():
                np_data = new_val.detach().cpu().to(torch_dtype).numpy().transpose()
                np_data *= scale_val if lora_name == "lora_B" else 1
                adapter_sets[adapter_name][new_key.replace(f".{adapter_name}", "")] = np_data

        # Save each LoRA set to disk
        for adapter_name, adapter_set in adapter_sets.items():
            output_path = save_weights(adapter_set, os.path.join(self.args.output, adapter_name), self.args.format)
            print(f"Exported {adapter_name} adapter weights to {output_path}")
