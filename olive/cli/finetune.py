# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser
from copy import deepcopy
from typing import ClassVar

from olive.cli.base import (
    BaseOliveCLICommand,
    add_dataset_options,
    add_input_model_options,
    add_logging_options,
    add_save_config_file_options,
    add_shared_cache_options,
    add_telemetry_options,
    get_input_model_config,
    update_dataset_options,
    update_shared_cache_options,
)
from olive.common.utils import set_nested_dict_value
from olive.telemetry import action


class FineTuneCommand(BaseOliveCLICommand):
    allow_unknown_args: ClassVar[bool] = True

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "finetune",
            help=(
                "Fine-tune a model on a dataset using peft. Huggingface training arguments can be provided along with"
                " the defined options."
            ),
        )

        # Model options
        add_input_model_options(sub_parser, enable_hf=True, default_output_path="finetuned-adapter")

        # LoRA options
        lora_group = sub_parser.add_argument_group("LoRA options")
        lora_group.add_argument(
            "--method",
            type=str,
            default="lora",
            choices=["lora", "qlora"],
            help="The method to use for fine-tuning",
        )
        lora_group.add_argument(
            "--lora_r",
            type=int,
            default=64,
            help="LoRA R value.",
        )
        lora_group.add_argument(
            "--lora_alpha",
            type=int,
            default=16,
            help="LoRA alpha value.",
        )
        # peft doesn't know about phi3, should we set it ourself in the lora pass based on model type?
        lora_group.add_argument(
            "--target_modules", type=str, help="The target modules for LoRA. If multiple, separate by comma."
        )

        sub_parser.add_argument(
            "--torch_dtype",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float16", "float32"],
            help="The torch dtype to use for training.",
        )

        add_dataset_options(sub_parser)
        add_shared_cache_options(sub_parser)
        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=FineTuneCommand)

    @action
    def run(self):
        return self._run_workflow()

    def parse_training_args(self) -> dict:
        if not self.unknown_args:
            return {}

        from transformers import HfArgumentParser, TrainingArguments

        arg_keys = {el[2:] for el in self.unknown_args if el.startswith("--")}
        parser = HfArgumentParser(TrainingArguments)
        # output_dir is required by the parser
        training_args = parser.parse_args(
            [*(["--output_dir", "dummy"] if "output_dir" not in arg_keys else []), *self.unknown_args]
        )

        return {k: v for k, v in vars(training_args).items() if k in arg_keys}

    def _get_run_config(self, tempdir: str) -> dict:
        input_model_config = get_input_model_config(self.args)
        assert input_model_config["type"].lower() == "hfmodel", "Only HfModel is supported in finetune command."

        finetune_key = ("passes", "f")
        to_replace = [
            ("input_model", input_model_config),
            ((*finetune_key, "type"), self.args.method),
            ((*finetune_key, "torch_dtype"), self.args.torch_dtype),
            ((*finetune_key, "training_args"), self.parse_training_args()),
            ((*finetune_key, "r"), self.args.lora_r),
            ((*finetune_key, "alpha"), self.args.lora_alpha),
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]
        if self.args.method == "lora" and self.args.target_modules:
            to_replace.append(((*finetune_key, "target_modules"), self.args.target_modules.split(",")))
        if self.args.method == "qlora":
            # bnb quant config is not needed, we only want the adapter
            to_replace.append(((*finetune_key, "save_quant_config"), False))

        config = deepcopy(TEMPLATE)
        update_dataset_options(self.args, config)

        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(config, keys, value)

        if self.args.eval_split:
            eval_data_config = deepcopy(config["data_configs"][0])
            eval_data_config["name"] = "eval_data"
            eval_data_config["load_dataset_config"]["split"] = self.args.eval_split
            config["data_configs"].append(eval_data_config)
            config["passes"]["f"]["eval_data_config"] = "eval_data"

        update_shared_cache_options(config, self.args)

        return config


TEMPLATE = {
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            # just a place holder, ep and device are not relevant for pytorch only workflow
            "accelerators": [{"device": "gpu", "execution_providers": ["CUDAExecutionProvider"]}],
        }
    },
    "data_configs": [
        {
            "name": "train_data",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {},
            "pre_process_data_config": {},
            "dataloader_config": {},
            "post_process_data_config": {},
        }
    ],
    "passes": {"f": {"train_data_config": "train_data"}},
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}
