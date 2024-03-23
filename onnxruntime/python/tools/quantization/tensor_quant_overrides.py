# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import json
from collections.abc import MutableMapping
from typing import Any

from .quant_utils import QuantType


class TensorQuantOverridesHelper(MutableMapping):
    """
    Utility wrapper over the tensor quantization overrides passed via extra_options.
    """

    def __init__(self, raw_overrides: dict[str, list[dict[str, Any]]]):
        self.overrides = raw_overrides
        self.quant_types = None

    def get_per_tensor_overrides(self, tensor_name: str) -> dict[str, Any]:
        overrides_list = self.overrides.get(tensor_name, [{}])
        num_overrides = len(overrides_list)
        if num_overrides > 1:
            raise ValueError(
                f"Expected tensor '{tensor_name}' to use per-tensor quantization overrides, "
                f"but found {num_overrides} per-channel overrides."
            )

        return overrides_list[0] if num_overrides > 0 else {}

    def get_per_channel_overrides(
        self,
        tensor_name: str,
        num_channels: int,
    ) -> list[dict[str, Any]]:
        overrides_list = self.overrides.get(tensor_name, [{} for i in range(num_channels)])

        if len(overrides_list) != num_channels:
            raise ValueError(
                f"Expected tensor '{tensor_name}' to have {num_channels} per-channel quantization overrides, "
                f"but found {len(overrides_list)} instead."
            )

        return overrides_list

    def get_quant_types(self) -> set[QuantType]:
        if self.quant_types is not None:
            return self.quant_types

        self.quant_types = set()

        if self.overrides:
            for quant_overrides_list in self.overrides.values():
                for quant_overrides in quant_overrides_list:
                    if "quant_type" in quant_overrides:
                        self.quant_types.add(quant_overrides["quant_type"])

                    if "convert" in quant_overrides and "quant_type" in quant_overrides["convert"]:
                        self.quant_types.add(quant_overrides["convert"]["quant_type"])

        return self.quant_types

    def is_valid(
        self,
        initializer_names: set[str],
        activation_names: set[str],
        default_activation_qtype,
    ) -> tuple[bool, str | None]:
        self.quant_types = set()

        # Validate that compatible/valid overrides are provided.
        if self.overrides:
            keys_unsupported_with_scale_zp = {"symmetric", "reduce_range", "rmax", "rmin"}

            for tensor_name, quant_overrides_list in self.overrides.items():
                if tensor_name not in initializer_names and tensor_name not in activation_names:
                    return False, f"Tensor '{tensor_name}' in TensorQuantOverrides is not present in the model"

                if not isinstance(quant_overrides_list, list):
                    return False, f"Tensor quantization overrides for '{tensor_name}' are not in a list"

                is_initializer = tensor_name in initializer_names
                if not is_initializer and len(quant_overrides_list) > 1:
                    return (
                        False,
                        f"Tensor '{tensor_name}' has a list of per-channel overrides, but is not an initializer",
                    )

                quant_type = None
                for index, quant_overrides in enumerate(quant_overrides_list):
                    if not isinstance(quant_overrides, dict):
                        return (
                            False,
                            f"Tensor quantization overrides at index {index} for '{tensor_name}' are not in a dict",
                        )

                    # For per-channel quantization, all channels must use the same quantization type.
                    # Therefore, if the user tries to override the quant_type for a channel, it must match in all
                    # other channels.
                    if index == 0:
                        quant_type = quant_overrides.get("quant_type")
                        if quant_type:
                            self.quant_types.add(quant_type)
                    elif quant_type != quant_overrides.get("quant_type"):
                        return (
                            False,
                            "Channel quantization types for tensor '{tensor_name}' do not match at index {index}.",
                        )

                    has_scale = "scale" in quant_overrides
                    has_zero_point = "zero_point" in quant_overrides

                    if (has_scale and not has_zero_point) or (has_zero_point and not has_scale):
                        return (
                            False,
                            "Must provide both 'scale' and 'zero_point' if one of the overrides is provided",
                        )

                    if has_scale:
                        for key in keys_unsupported_with_scale_zp:
                            if key in quant_overrides:
                                return (
                                    False,
                                    f"Tensor override option '{key}' is invalid with 'scale' and 'zero_point'",
                                )

                    if "reduce_range" in quant_overrides and not is_initializer:
                        return (
                            False,
                            f"Option 'reduce_range' is only supported for initializers, not for activation {tensor_name}",
                        )

                    if "convert" in quant_overrides:
                        if index > 0:
                            return (
                                False,
                                f"Per-channel overrides (tensor '{tensor_name}') do not support 'convert'.",
                            )

                        if is_initializer:
                            return False, "Cannot use 'convert' override for initializers"

                        if "quant_type" not in quant_overrides["convert"]:
                            return False, f"'convert' options (tensor '{tensor_name}') must specify a 'quant_type'"

                        if "reduce_range" in quant_overrides["convert"]:
                            return (
                                False,
                                f"Option 'reduce_range' is only supported for initializers, not for activation {tensor_name}",
                            )

                        convert_quant_type = quant_overrides["convert"]["quant_type"]
                        original_quant_type = quant_type if quant_type is not None else default_activation_qtype
                        if convert_quant_type == original_quant_type:
                            return (
                                False,
                                f"'convert' quant_type must differ from original quant_type (tensor '{tensor_name}')",
                            )

                        convert_has_scale = "scale" in quant_overrides["convert"]
                        convert_has_zero_point = "zero_point" in quant_overrides["convert"]

                        if (convert_has_scale and not convert_has_zero_point) or (
                            convert_has_zero_point and not convert_has_scale
                        ):
                            return (
                                False,
                                f"Must provide both 'scale' and 'zero_point' if one of the overrides is provided (tensor '{tensor_name}')",
                            )

                        if convert_has_scale:
                            for key in keys_unsupported_with_scale_zp:
                                if key in quant_overrides["convert"]:
                                    return (
                                        False,
                                        f"Tensor override option '{key}' is invalid with 'scale' and 'zero_point' (tensor '{tensor_name}')",
                                    )

                        self.quant_types.add(convert_quant_type)

        return True, None

    def pprint_str(self, indent=None) -> str:
        return json.dumps(self.overrides, default=str, indent=indent)

    def get_dict(self) -> dict[str, list[dict[str, Any]]]:
        return self.overrides

    # Required implementations of abstract methods in collections.abc.MutableMapping
    # so that this class can be used like a dict.
    def __setitem__(self, key: str, value: list[dict]):
        self.overrides[key] = value

    def __getitem__(self, key: str) -> list[dict]:
        return self.overrides[key]

    def __delitem__(self, key: str):
        del self.overrides[key]

    def __iter__(self):
        return iter(self.overrides)

    def __len__(self):
        return len(self.overrides)

    def __str__(self) -> str:
        return str(self.overrides)

    def __repr__(self) -> str:
        return f"{super().__repr__()}, TensorQuantOverridesHelper({self.overrides})"
