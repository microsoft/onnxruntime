# -----------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -----------------------------------------------------------------------------
from argparse import ArgumentParser
from copy import deepcopy

from olive.cli.base import (
    BaseOliveCLICommand,
    add_input_model_options,
    add_logging_options,
    add_save_config_file_options,
    add_shared_cache_options,
    add_telemetry_options,
    get_input_model_config,
    update_shared_cache_options,
)
from olive.common.utils import set_nested_dict_value
from olive.telemetry import action


class BenchmarkCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser("benchmark", help="Evaluate the model using lm-eval.")

        # model options
        add_input_model_options(
            sub_parser, enable_hf=True, enable_hf_adapter=True, enable_pt=True, default_output_path="onnx-model"
        )

        # lm-eval options
        lmeval_group = sub_parser.add_argument_group("lm-eval evaluator options")
        lmeval_group.add_argument(
            "--tasks",
            type=str,
            required=True,
            nargs="*",
            help="List of tasks to evaluate on.",
        )

        lmeval_group.add_argument(
            "--device",
            type=str,
            default="cpu",
            choices=["cpu", "gpu"],
            help="Target device for evaluation.",
        )

        lmeval_group.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Batch size.",
        )

        lmeval_group.add_argument(
            "--max_length",
            type=int,
            default=1024,
            help="Maximum length of input + output.",
        )

        lmeval_group.add_argument(
            "--limit",
            type=float,
            default=1,
            help="Number (or percentage of dataset) of samples to use for evaluation.",
        )

        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_shared_cache_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=BenchmarkCommand)

    @action
    def run(self):
        return self._run_workflow()

    def _get_run_config(self, tempdir: str) -> dict:
        config = deepcopy(TEMPLATE)

        input_model_config = get_input_model_config(self.args)
        assert input_model_config["type"].lower() in {
            "hfmodel",
            "pytorchmodel",
            "onnxmodel",
        }, "Only HfModel, PyTorchModel and OnnxModel are supported in benchmark command."

        to_replace = [
            ("input_model", input_model_config),
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
            (("systems", "local_system", "accelerators", 0, "device"), self.args.device),
            (
                ("systems", "local_system", "accelerators", 0, "execution_providers"),
                [("CUDAExecutionProvider" if self.args.device == "gpu" else "CPUExecutionProvider")],
            ),
            (("evaluators", "evaluator", "tasks"), self.args.tasks),
            (("evaluators", "evaluator", "device"), self.args.device),
            (("evaluators", "evaluator", "batch_size"), self.args.batch_size),
            (("evaluators", "evaluator", "max_length"), self.args.max_length),
            (("evaluators", "evaluator", "device"), self.args.device),
            (("evaluators", "evaluator", "limit"), self.args.limit),
        ]

        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(config, keys, value)
        update_shared_cache_options(config, self.args)

        return config


TEMPLATE = {
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "evaluators": {
        "evaluator": {
            "type": "LMEvaluator",
            "tasks": [],
            "batch_size": 16,
            "max_length": 1024,
            "device": "cpu",
            "limit": 64,
        }
    },
    "evaluator": "evaluator",
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}
