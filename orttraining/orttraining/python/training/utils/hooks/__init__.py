# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

__all__ = ["StatisticsSubscriber", "GlobalSubscriberManager", "inspect_activation"]

import torch

from ._statistics_subscriber import StatisticsSubscriber
from ._subscriber_manager import SubscriberManager

# Define a global uninitialized subscriber manager for usage where it is needed by different Python files.
GlobalSubscriberManager = SubscriberManager()


def inspect_activation(activation_name: str, tensor: torch.Tensor) -> torch.Tensor:
    """
    A helper function to inspect the activation tensor during the forward path.
    """
    statistic_subscriber = None
    for subscriber in GlobalSubscriberManager.get_run_context().global_states.subscribers:
        if isinstance(subscriber, StatisticsSubscriber):
            statistic_subscriber = subscriber
            break

    if statistic_subscriber is None:
        raise RuntimeError("No statistics subscriber found, please make sure the statistics subscriber is registered.")

    return statistic_subscriber.inspect_adhoc_activation(
        activation_name, tensor, GlobalSubscriberManager.get_run_context()
    )
