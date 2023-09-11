# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import torch

__all__ = [
    "StatisticsSubscriber",
    "GlobalSubscriberManager",
    "inspect_activation",
    "ZeROOffloadSubscriber",
    "configure_ort_compatible_zero_stage3",
]

from ._statistics_subscriber import StatisticsSubscriber, _InspectActivation
from ._subscriber_manager import SubscriberManager
from ._zero_offload_subscriber import ZeROOffloadSubscriber, configure_ort_compatible_zero_stage3

# Define a global uninitialized subscriber manager for usage where it is needed by different Python files.
GlobalSubscriberManager = SubscriberManager()


def inspect_activation(activation_name: str, tensor: torch.Tensor) -> torch.Tensor:
    for sub in GlobalSubscriberManager._subscribers:
        if isinstance(sub, StatisticsSubscriber):
            return _InspectActivation.apply(
                activation_name,
                None,
                GlobalSubscriberManager.get_run_context(),
                tensor,
                sub.module_post_forward_impl,
                sub.module_pre_backward_impl,
            )

    raise RuntimeError("StatisticsSubscriber is not registered, cannot inspect activation.")
