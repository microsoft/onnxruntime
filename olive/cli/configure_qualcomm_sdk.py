# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser

from olive.cli.base import BaseOliveCLICommand, add_telemetry_options
from olive.telemetry import action


class ConfigureQualcommSDKCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "configure-qualcomm-sdk",
            help="Configure Qualcomm SDK for Olive",
        )
        sub_parser.add_argument(
            "--py_version",
            type=str,
            help="Python version: Use 3.6 for tensorflow 1.15 and 3.8 otherwise",
            required=True,
            choices=["3.6", "3.8"],
        )
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=ConfigureQualcommSDKCommand)

    @action
    def run(self):
        from olive.platform_sdk.qualcomm.configure.configure import configure

        configure(self.args.py_version)
