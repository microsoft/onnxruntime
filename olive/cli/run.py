# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser

from olive.cli.base import (
    BaseOliveCLICommand,
    add_input_model_options,
    add_logging_options,
    add_telemetry_options,
    get_input_model_config,
)
from olive.telemetry import action


class WorkflowRunCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser("run", help="Run an olive workflow")
        sub_parser.add_argument("--run-config", "--config", type=str, help="Path to json config file", required=True)
        sub_parser.add_argument(
            "--list_required_packages", help="List packages required to run the workflow", action="store_true"
        )
        sub_parser.add_argument(
            "--tempdir", type=str, help="Root directory for tempfile directories and files", required=False
        )
        sub_parser.add_argument(
            "--package-config",
            type=str,
            required=False,
            help=(
                "For advanced users. Path to optional package (json) config file with location "
                "of individual pass module implementation and corresponding dependencies. "
                "Configuration might also include user owned/proprietary/private pass implementations."
            ),
        )
        add_logging_options(sub_parser, default=None)
        add_input_model_options(
            sub_parser.add_argument_group("Model options (not required)"),
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_onnx=True,
            required=False,
        )
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=WorkflowRunCommand)

    @action
    def run(self):
        from olive.common.config_utils import load_config_file
        from olive.workflows import run as olive_run

        # allow the run_config to be a dict already (for api use)
        run_config = self.args.run_config
        if not isinstance(run_config, dict):
            run_config = load_config_file(run_config)
        if input_model_config := get_input_model_config(self.args, required=False):
            print("Replacing input model config in run config")
            run_config["input_model"] = input_model_config

        for arg_key, rc_key in [("output_path", "output_dir"), ("log_level", "log_severity_level")]:
            if (arg_value := getattr(self.args, arg_key)) is not None:
                print(f"Replacing {rc_key} in run config with {arg_value}")
                # remove value from engine config if it exists
                run_config.get("engine", {}).pop(rc_key, None)
                # add value to run config directly
                run_config[rc_key] = arg_value

        workflow_output = olive_run(
            run_config,
            list_required_packages=self.args.list_required_packages,
            tempdir=self.args.tempdir,
            package_config=self.args.package_config,
        )

        if self.args.list_required_packages is True:
            print("Required packages listed!")

        return workflow_output
