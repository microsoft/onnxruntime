# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

__all__ = [
    "StatisticsSubscriber",
    "GlobalSubscriberManager",
    "_InspectActivation",
]

from ._statistics_subscriber import StatisticsSubscriber
from ._subscriber_manager import SubscriberManager, _InspectActivation

# Define a global uninitialized subscriber manager for usage where it is needed by different Python files.
GlobalSubscriberManager = SubscriberManager()
