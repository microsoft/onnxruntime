# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any

from olive.cli.base import (
    BaseOliveCLICommand,
    add_accelerator_options,
    add_input_model_options,
    add_logging_options,
    add_save_config_file_options,
    add_telemetry_options,
    get_input_model_config,
    update_accelerator_options,
)
from olive.telemetry import action


@action
class RunPassCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "run-pass",
            help="Run a single pass on the input model (supports HuggingFace, ONNX, PyTorch, and Azure ML models)",
        )

        # Pass selection
        sub_parser.add_argument(
            "--pass-name",
            type=str,
            help="Name of the pass to run on the input model.",
        )

        # Pass configuration options (common parameters)
        sub_parser.add_argument(
            "--pass-config",
            type=str,
            help="JSON string with pass-specific configuration parameters.",
        )

        # List available passes
        sub_parser.add_argument(
            "--list-passes",
            action="store_true",
            help="List all available passes and exit.",
        )

        # Model options
        add_input_model_options(
            sub_parser,
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_onnx=True,
            default_output_path="run-pass-output",
            required=False,  # Make model optional to allow --list-passes
        )

        # Accelerator options
        add_accelerator_options(sub_parser)

        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=RunPassCommand)

    def _get_run_config(self, tempdir: str) -> dict[str, Any]:
        # Import these only when needed to avoid circular imports
        from olive.common.utils import set_nested_dict_value
        from olive.package_config import OlivePackageConfig

        config = deepcopy(TEMPLATE)

        # Set input model from args
        config["input_model"] = get_input_model_config(self.args)

        # Set the single pass configuration
        pass_name = self.args.pass_name

        # Validate that the pass exists
        olive_config = OlivePackageConfig.load_default_config()
        try:
            olive_config.get_pass_module_config(pass_name)
        except ValueError as verror:
            available_passes = list(olive_config.passes.keys())
            raise ValueError(
                f"Pass '{pass_name}' not found. Available passes: {', '.join(available_passes)}"
            ) from verror

        # Create a simple pass configuration
        pass_config = {"type": pass_name}

        # Add pass-specific configuration if provided
        if hasattr(self.args, "pass_config") and self.args.pass_config:
            import json

            try:
                additional_config = json.loads(self.args.pass_config)
                pass_config.update(additional_config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in --pass-config: {e}") from e

        config["passes"] = {pass_name.lower(): pass_config}

        # Customize the config for user choices
        to_replace = [
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]
        for k, v in to_replace:
            if v is not None:
                set_nested_dict_value(config, k, v)

        # Update accelerator options (device, execution provider, memory)
        update_accelerator_options(self.args, config)

        # Ensure provider and device are consistent
        self._ensure_device_provider_consistency(config)

        return config

    def run(self):
        # Check if user wants to list passes
        if hasattr(self.args, "list_passes") and self.args.list_passes:
            self._list_passes()
            return None

        # Validate required arguments when not listing passes
        if not self.args.pass_name:
            raise ValueError("--pass-name is required when not using --list-passes")

        if not getattr(self.args, "model_name_or_path", None):
            raise ValueError("-m/--model_name_or_path is required when not using --list-passes")

        return self._run_workflow()

    def _ensure_device_provider_consistency(self, config):
        """Ensure device and provider are consistent."""
        from olive.hardware.constants import DEVICE_TO_EXECUTION_PROVIDERS, ExecutionProvider

        # Get the current accelerator config
        accelerator = config["systems"]["local_system"]["accelerators"][0]
        providers = accelerator.get("execution_providers", [])
        current_device = accelerator.get("device", "cpu")

        if not providers:
            return

        provider = providers[0]  # Take the first provider

        # Define provider-specific device preferences
        # For providers that can run on multiple devices, we choose the most appropriate one
        provider_device_preference = {
            ExecutionProvider.CPUExecutionProvider: "cpu",
            ExecutionProvider.CUDAExecutionProvider: "gpu",
            ExecutionProvider.ROCMExecutionProvider: "gpu",
            ExecutionProvider.TensorrtExecutionProvider: "gpu",
            ExecutionProvider.NvTensorRTRTXExecutionProvider: "gpu",
            ExecutionProvider.MIGraphXExecutionProvider: "gpu",
            ExecutionProvider.JsExecutionProvider: "gpu",
            ExecutionProvider.DmlExecutionProvider: "gpu",  # Prefer GPU for DirectML
            ExecutionProvider.QNNExecutionProvider: "npu",
            ExecutionProvider.VitisAIExecutionProvider: "npu",
            ExecutionProvider.OpenVINOExecutionProvider: "cpu",  # Prefer CPU for OpenVINO (more common)
        }

        # Check if current device is valid for the provider
        valid_devices = []
        for device, device_providers in DEVICE_TO_EXECUTION_PROVIDERS.items():
            if provider in device_providers:
                valid_devices.append(device)

        if current_device not in valid_devices and valid_devices:
            # Current device is not valid for the provider, use the preferred device
            preferred_device = provider_device_preference.get(provider, valid_devices[0])
            accelerator["device"] = preferred_device
            print(f"Note: Setting device to '{preferred_device}' to match provider '{provider}'")

    def _list_passes(self):
        """List all available passes."""
        from olive.package_config import OlivePackageConfig

        try:
            olive_config = OlivePackageConfig.load_default_config()
            available_passes = sorted(olive_config.passes.keys())

            print("Available passes:")
            for i, pass_name in enumerate(available_passes, 1):
                print(f"{i:3d}. {pass_name}")

            print(f"\nTotal: {len(available_passes)} passes available")
            print("\nUsage: olive run-pass --pass-name <PassName> -m <model> -o <output>")

        except Exception as e:
            print(f"Error loading pass configurations: {e}")
            print("Unable to list available passes.")


# Template configuration for the one command
TEMPLATE = {
    "input_model": {"type": "HfModel", "load_kwargs": {"attn_implementation": "eager"}},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "output_dir": "models",
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}
