# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _configuration_option.py

from abc import ABC, abstractmethod

class ConfigurationOption(ABC):
    """All configurable options should inherit from this base class."""

    @abstractmethod
    def configure(self):
        pass

    @abstractmethod
    def reset(self):
        pass
