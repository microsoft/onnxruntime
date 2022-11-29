# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _provider_type.py

from typing import NewType, Text


class DeviceType(object):
    _Type = NewType("_Type", int)
    cpu = _Type(0)  # type: _Type
    cuda = _Type(1)  # type: _Type


class ProviderType(object):
    """
    Describes device type and device id
    syntax: device_type:device_id(optional)
    example: 'cpu', 'cuda', 'cuda:1'
    """

    def __init__(self, device):  # type: (Text) -> None
        options = device.split(":")
        self.type = getattr(DeviceType, options[0])
        self.type_str = options[0]
        self.device_id = 0
        if len(options) > 1:
            self.device_id = int(options[1])
