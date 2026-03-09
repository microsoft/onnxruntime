# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from olive.cli.base import BaseOliveCLICommand, add_telemetry_options
from olive.common.container_client_factory import AzureContainerClientFactory
from olive.telemetry import action

logger = logging.getLogger(__name__)


class SharedCacheCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser):
        sub_parser = parser.add_parser("shared-cache", help="Shared cache model operations")
        sub_parser.add_argument(
            "--delete",
            action="store_true",
            help="Delete a model cache from the shared cache.",
        )
        sub_parser.add_argument(
            "--all",
            action="store_true",
            help="Delete all model cache from the cloud cache.",
        )
        sub_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Confirm the deletion without prompting for confirmation.",
        )
        sub_parser.add_argument(
            "--account",
            type=str,
            required=True,
            help="The account name for the shared cache.",
        )
        sub_parser.add_argument(
            "--container",
            type=str,
            required=True,
            help="The container name for the shared cache.",
        )
        sub_parser.add_argument(
            "--model_hash",
            type=str,
            help="The model hash to remove from the shared cache.",
        )
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=SharedCacheCommand)

    @action
    def run(self):
        container_client_factory = AzureContainerClientFactory(self.args.account, self.args.container)
        if self.args.delete:
            if self.args.all:
                if self.args.yes:
                    container_client_factory.delete_all()
                else:
                    confirm = input("Are you sure you want to delete all cache? (y/n): ")
                    if confirm.lower() == "y":
                        container_client_factory.delete_all()
            else:
                container_client_factory.delete_blob(self.args.model_hash)
