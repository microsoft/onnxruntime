# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from argparse import ArgumentParser
from warnings import warn

from olive.cli.auto_opt import AutoOptCommand
from olive.cli.benchmark import BenchmarkCommand
from olive.cli.capture_onnx import CaptureOnnxGraphCommand
from olive.cli.configure_qualcomm_sdk import ConfigureQualcommSDKCommand
from olive.cli.convert_adapters import ConvertAdaptersCommand
from olive.cli.diffusion_lora import DiffusionLoraCommand
from olive.cli.extract_adapters import ExtractAdaptersCommand
from olive.cli.finetune import FineTuneCommand
from olive.cli.generate_adapter import GenerateAdapterCommand
from olive.cli.generate_cost_model import GenerateCostModelCommand
from olive.cli.optimize import OptimizeCommand
from olive.cli.quantize import QuantizeCommand
from olive.cli.run import WorkflowRunCommand
from olive.cli.run_pass import RunPassCommand
from olive.cli.session_params_tuning import SessionParamsTuningCommand
from olive.cli.shared_cache import SharedCacheCommand
from olive.telemetry import Telemetry


def get_cli_parser(called_as_console_script: bool = True) -> ArgumentParser:
    """Get the CLI parser for Olive.

    :param called_as_console_script: Whether the script was called as a console script.
    :return: The CLI parser.
    """
    parser = ArgumentParser("Olive CLI tool", usage="olive" if called_as_console_script else "python -m olive")
    commands_parser = parser.add_subparsers()

    # Register commands
    # TODO(jambayk): Consider adding a common tempdir option to all commands
    # NOTE: The order of the commands is to organize the documentation better.
    WorkflowRunCommand.register_subcommand(commands_parser)
    RunPassCommand.register_subcommand(commands_parser)
    AutoOptCommand.register_subcommand(commands_parser)
    OptimizeCommand.register_subcommand(commands_parser)
    CaptureOnnxGraphCommand.register_subcommand(commands_parser)
    DiffusionLoraCommand.register_subcommand(commands_parser)
    FineTuneCommand.register_subcommand(commands_parser)
    GenerateAdapterCommand.register_subcommand(commands_parser)
    ConvertAdaptersCommand.register_subcommand(commands_parser)
    QuantizeCommand.register_subcommand(commands_parser)
    SessionParamsTuningCommand.register_subcommand(commands_parser)
    GenerateCostModelCommand.register_subcommand(commands_parser)
    ConfigureQualcommSDKCommand.register_subcommand(commands_parser)
    SharedCacheCommand.register_subcommand(commands_parser)
    ExtractAdaptersCommand.register_subcommand(commands_parser)
    BenchmarkCommand.register_subcommand(commands_parser)

    return parser


def main(raw_args=None, called_as_console_script: bool = True):
    parser = get_cli_parser(called_as_console_script)

    args, unknown_args = parser.parse_known_args(raw_args)

    telemetry = Telemetry()
    if args.disable_telemetry:
        telemetry.disable_telemetry()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Run the command
    service = args.func(parser, args, unknown_args)
    service.run()
    telemetry.shutdown()


def legacy_call(deprecated_module: str, command_name: str, *args):
    """Run a command with a warning about the deprecation of the module.

    Command arguments are taken from the command line.

    :param deprecated_module: The deprecated module name.
    :param command_name: The command name to run.
    :param args: Additional arguments to pass to the command.
    """
    warn(
        f"Running `python -m {deprecated_module}` is deprecated and might be removed in the future. Please use"
        f" `olive {command_name}` or `python -m olive {command_name}` instead.",
        FutureWarning,
    )

    # get args from command line
    raw_args = [command_name, *args]
    main(raw_args)


if __name__ == "__main__":
    main(called_as_console_script=False)
