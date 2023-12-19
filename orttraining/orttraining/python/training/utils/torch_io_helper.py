# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
import warnings
from collections import abc
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import torch
import numpy

from onnxruntime.training.utils.torch_profile_utils import nvtx_function_decorator


class PrimitiveType:
    """Helper class for Python primitive types."""

    _primitive_types = {int, bool, float}  # noqa: RUF012

    @staticmethod
    def is_primitive_type(value):
        """Check if `value` is a Python primitive type."""
        return type(value) in PrimitiveType._primitive_types

    @staticmethod
    def get_tensor(value, device) -> torch.Tensor:
        """Convert `value` to a torch.Tensor."""
        return torch.tensor(value, device=device)

    @staticmethod
    def get_primitive_dtype(value):
        # If `value` is a boolean, save the value of the boolean in dtype.
        # This way, if the value changes from one forward call to the next, the schema will mismatch,
        # and the model will be re-exported.
        return f"{type(value)!s}_{value}" if isinstance(value, bool) else str(type(value))


# Data types supported as model inputs and outputs.
ORTModelInputOutputType = Union[
    None,
    str,
    int,
    bool,
    float,
    numpy.ndarray,
    torch.Tensor,
    Sequence["ORTModelInputOutputType"],
    Mapping[str, "ORTModelInputOutputType"],
]


class _TensorStub:
    """Tensor stub class used to represent model's input or output"""

    __slots__ = ["tensor_idx", "name", "dtype", "shape", "shape_dims"]

    def __init__(
        self,
        tensor_idx: int,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        shape=None,
        shape_dims: Optional[int] = None,
    ):
        self.tensor_idx = tensor_idx
        self.name: Optional[str] = name
        self.dtype: Optional[str] = dtype
        self.shape = shape
        self.shape_dims: Optional[int] = shape_dims  # r.g. rank.

    def __repr__(self) -> str:
        result = "_TensorStub("

        list_of_str = []
        if self.tensor_idx is not None:
            list_of_str.append(f"tensor_idx={self.tensor_idx}")
        if self.name is not None:
            list_of_str.append(f"name={self.name}")
        if self.dtype is not None:
            list_of_str.append(f"dtype={self.dtype}")
        if self.shape is not None:
            list_of_str.append(f"shape={self.shape}")
        if self.shape_dims is not None:
            list_of_str.append(f"shape_dims={self.shape_dims}")

        result += ",".join(list_of_str) + ")"
        return result

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        if not other:
            return False
        elif not isinstance(other, _TensorStub):
            raise NotImplementedError("_TensorStub must only be compared to another _TensorStub instance!")
        elif self.tensor_idx != other.tensor_idx:
            return False
        elif self.name != other.name:
            return False
        elif self.dtype != other.dtype:
            return False
        elif self.shape != other.shape:
            return False
        elif self.shape_dims != other.shape_dims:
            return False
        return True


# Data schema used to represent model's input or output.
ORTModelInputOutputSchemaType = Union[
    None,
    str,
    numpy.ndarray,
    _TensorStub,
    Sequence["ORTModelInputOutputSchemaType"],
    Mapping[str, "ORTModelInputOutputSchemaType"],
]


def _warn_of_constant_inputs(data):
    warnings.warn(
        f"Received input of type {type(data)} which may be treated as a constant by ORT by default."
        " Please consider moving constant arguments to the model constructor."
    )


@nvtx_function_decorator
def extract_data_and_schema(
    data: ORTModelInputOutputType, constant_as_tensor=False, device: Optional[torch.device] = None
) -> Tuple[List[torch.Tensor], ORTModelInputOutputSchemaType]:
    """Extract the data schema by replacing every torch.Tensor value with _TensorStub, and return all tensors in
    a list.

    Depth first traversal to iterate over the data:
    > Replace every tensor with a stub
    > Replace None/str typed data with itself
    > For other primitive types:
        If constant_as_tensor is True, create tensor from data , and replace them with a stub in returned schema;
        Otherwise, replace them with themselves in returned schema.

    Examples:
        Example 1, list:
            data = [torch.tensor(1), torch.tensor(2)]
            schema = [_TensorStub(shape=()), _TensorStub(shape=())]

        Example 2, dict:
            data = {"a": torch.tensor(1), "b": torch.tensor(2)}
            schema = {"a": _TensorStub(shape=()), "b": _TensorStub(shape=())}

        Example 3, dict of list:
            data = {"a": [torch.tensor(1), torch.tensor(2)], "b": [torch.tensor(3), torch.tensor(4)]}
            schema = {"a": [_TensorStub(shape=()), _TensorStub(shape=())],
                        "b": [_TensorStub(shape=()), _TensorStub(shape=())]}

        Example 4, nested dict:
            data = {"a": {"b": torch.tensor(1), "c": torch.tensor(2)},
                    "d": {"e": torch.tensor(3), "f": torch.tensor(4)}}
            schema = {"a": {"b": _TensorStub(shape=()), "c": _TensorStub(shape=())},
                        "d": {"e": _TensorStub(shape=()), "f": _TensorStub(shape=())}}

        Example 5, dict of mixed list and dict:
            data = {"a": [torch.tensor(1), torch.tensor(2)], "b": {"c": torch.tensor(3), "d": torch.tensor(4)}}
            schema = {"a": [_TensorStub(shape=()), _TensorStub(shape=())],
                        "b": {"c": _TensorStub(shape=()), "d": _TensorStub(shape=())}}

    Args:
        data: The data to extract the schema from, which can be in any kind of nested structure,
            including Sequence and Mapping.
        constant_as_tensor: Whether to treat constant inputs as tensor. Default is False.
        device: The device to create tensor for the constant.


    Returns:
        Tuple:
            The first value: a list of tensors extracted from the data.
            The second value: schema of the data, which has the same structure as the data.


    """

    flatten_tensor_data = []
    tensor_idx = [-1]

    def _flatten_from_data(data: ORTModelInputOutputType, prefix_name: str = ""):
        if data is None:
            return data
        elif isinstance(data, str):
            _warn_of_constant_inputs(data)
            return data
        elif PrimitiveType.is_primitive_type(data):
            _warn_of_constant_inputs(data)
            if constant_as_tensor:
                tensor_idx[0] += 1
                flatten_tensor_data.append(PrimitiveType.get_tensor(data, device))
                return _TensorStub(
                    tensor_idx[0], dtype=PrimitiveType.get_primitive_dtype(data), shape_dims=0, name=prefix_name
                )
            return data
        elif isinstance(data, numpy.ndarray):
            _warn_of_constant_inputs(data)
            return data
        # Depth first traversal to iterate over the data to replace every tensor with a stub
        elif isinstance(data, torch.Tensor):
            tensor_idx[0] += 1
            flatten_tensor_data.append(data)
            return _TensorStub(
                tensor_idx[0],
                dtype=str(data.dtype),
                shape_dims=len(data.size()),
                name=prefix_name,
            )
        elif isinstance(data, abc.Sequence):
            sequence_type = type(data)
            stubbed_schema = []
            for i, val in enumerate(data):
                stubbed_schema.append(_flatten_from_data(val, f"{prefix_name}_{i}" if prefix_name else f"{i}"))

            try:
                # namedtuple can be created by passing the list sequence to method _make
                stubbed_schema = sequence_type._make(stubbed_schema)
            except AttributeError:
                # If attribute error is encountered, create the sequence directly
                stubbed_schema = sequence_type(stubbed_schema)
            return stubbed_schema
        elif isinstance(data, abc.Mapping):
            dict_type = type(data)
            stubbed_schema = {}
            for key, val in sorted(data.items()):
                stubbed_schema[key] = _flatten_from_data(val, f"{prefix_name}_{key}" if prefix_name else f"{key}")
            stubbed_schema = dict_type(**stubbed_schema)
            return stubbed_schema
        else:
            raise TypeError(f"Unsupported flatten data type: {type(data)}")

    schemas = _flatten_from_data(data)
    return flatten_tensor_data, schemas


@nvtx_function_decorator
def unflatten_data_using_schema(
    data: List[torch.Tensor], schema: ORTModelInputOutputSchemaType
) -> ORTModelInputOutputType:
    """Follows the schema to generate an output that is expected by the user.

    Depth first traversal to iterate over the schema:
    > Replace every _TensorStub with a tensor from data. For each _TensorStub, the tensor is selected from `data`
      by its tensor_idx.

    Examples:
        Example 1, schema is a list:
            data = [torch.tensor(1), torch.tensor(2)]
            schema = [_TensorStub(shape=()), True, _TensorStub(shape=()), 1, 2.0, "abc"]
            output = [torch.tensor(1), True, torch.tensor(2), 1, 2.0, "abc"]

        Example 2, schema is a dict:
            data = [torch.tensor(1), torch.tensor(2)]
            schema = {"a": _TensorStub(shape=()), "b": _TensorStub(shape=()), "c": True, "d": 1, "e": 2.0, "f": "abc"}
            output = {"a": torch.tensor(1), "b": torch.tensor(2), "c": True, "d": 1, "e": 2.0, "f": "abc"}

        Example 3, schema is a dict of list:
            data = [torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(4)]
            schema = {"a": [_TensorStub(shape=()), _TensorStub(shape=())], "b": [_TensorStub(shape=()), _TensorStub(shape=())]}
            output = {"a": [torch.tensor(1), torch.tensor(2)], "b": [torch.tensor(3), torch.tensor(4)]}

        Example 4, schema is a nested dict:
            data = [torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(4)]
            schema = {"a": {"b": _TensorStub(shape=()), "c": _TensorStub(shape=())},
                        "d": {"e": _TensorStub(shape=()), "f": _TensorStub(shape=())}}
            output = {"a": {"b": torch.tensor(1), "c": torch.tensor(2)}, "d": {"e": torch.tensor(3), "f": torch.tensor(4)}}

        Example 5, dict of mixed list and dict:
            data = [torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(4)]
            schema = {"a": [_TensorStub(shape=()), _TensorStub(shape=())], "b": {"c": _TensorStub(shape=()), "d": _TensorStub(shape=())}}
            output = {"a": [torch.tensor(1), torch.tensor(2)], "b": {"c": torch.tensor(3), "d": torch.tensor(4)}}


    Args:
        data (List[torch.Tensor]): List of tensors to be used to replace the _TensorStub in schema
        schema (ORTModelInputOutputSchemaType): Schema to follow to generate the output

    Returns:
        output (ORTModelInputOutputType): Output that is expected by the user

    """

    def _replace_stub_with_tensor_value(data_schema: ORTModelInputOutputSchemaType, data: List[torch.Tensor]):
        # Recursively traverse across user_output and replace all _TensorStub
        # with torch.Tensor values from outputs following output_idx

        if data_schema is None:
            return None
        elif isinstance(data_schema, str):
            return data_schema
        elif PrimitiveType.is_primitive_type(data_schema):
            return data_schema
        elif isinstance(data_schema, numpy.ndarray):
            return data_schema
        elif isinstance(data_schema, _TensorStub):
            assert isinstance(
                data[data_schema.tensor_idx], torch.Tensor
            ), f"Expecting torch.Tensor, got {type(data[data_schema.tensor_idx])}"
            return data[data_schema.tensor_idx]
        elif isinstance(data_schema, abc.Sequence):
            sequence_type = type(data_schema)
            if hasattr(sequence_type, "_make"):  # namedtuple
                sequence_type = type(data_schema)
                data_schema = sequence_type._make(_replace_stub_with_tensor_value(uo, data) for uo in data_schema)
            else:
                data_schema = sequence_type(_replace_stub_with_tensor_value(uo, data) for uo in data_schema)
            return data_schema
        elif isinstance(data_schema, abc.Mapping):
            new_user_output = copy.copy(data_schema)
            for key, schema_val in sorted(data_schema.items()):
                new_user_output[key] = _replace_stub_with_tensor_value(schema_val, data)
            data_schema = new_user_output

            return data_schema

        raise TypeError(f"Unsupported unflatten data type: {type(data_schema)}.")

    user_output = _replace_stub_with_tensor_value(schema, data)
    return user_output
