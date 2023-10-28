# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import abc

import pytest
import torch

from onnxruntime.training.utils import extract_data_and_schema, unflatten_data_using_schema
from onnxruntime.training.utils.torch_io_helper import _TensorStub


@pytest.mark.parametrize(
    "input_output_map",
    [
        # single element
        [
            True,  # test input
            [],  # expected output: flatten tensor list
            True,  # expected output: extracted schema
            # expected output: flatten tensor list when constant_as_tensor=True
            [torch.tensor(True)],
        ],
        [
            False,  # test input
            [],  # expected output: flatten tensor list
            False,  # expected output: extracted schema
            # expected output: flatten tensor list when constant_as_tensor=True
            [torch.tensor(False)],
        ],
        [
            1,  # test input
            [],  # expected output: flatten tensor list
            1,  # expected output: extracted schema
            # expected output: flatten tensor list when constant_as_tensor=True
            [torch.tensor(1)],
        ],
        [
            2.0,  # test input
            [],  # expected output: flatten tensor list
            2.0,  # expected output: extracted schema
            # expected output: flatten tensor list when constant_as_tensor=True
            [torch.tensor(2.0)],
        ],
        [
            "abc",  # test input
            [],  # expected output: flatten tensor list
            "abc",  # expected output: extracted schema
            # expected output: flatten tensor list when constant_as_tensor=True
            [],
        ],
        [
            None,  # test input
            [],  # expected output: flatten tensor list
            None,  # expected output: extracted schema
            # expected output: flatten tensor list when constant_as_tensor=True
            [],
        ],
        [
            torch.tensor([1, 2, 3]),  # test input
            [torch.tensor([1, 2, 3])],  # expected output: flatten tensor list
            _TensorStub(tensor_idx=0, name="", dtype=torch.int64, shape_dims=1),  # expected output: extracted schema
            # expected output: flatten tensor list when constant_as_tensor=True
            [torch.tensor([1, 2, 3])],
        ],
        # list
        [
            [True, False, 1, 2.0, "abc", None],  # test input
            [],  # expected output: flatten tensor list
            [True, False, 1, 2.0, "abc", None],  # expected output: extracted schema
            # expected output: flatten tensor list when constant_as_tensor=True
            [torch.tensor(True), torch.tensor(False), torch.tensor(1), torch.tensor(2.0)],
        ],
        [
            [True, False, 1, 2.0, "abc", None, torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
            [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
            [
                True,
                False,
                1,
                2.0,
                "abc",
                None,
                _TensorStub(tensor_idx=0, name="6", dtype=torch.int64, shape_dims=1),
                _TensorStub(tensor_idx=1, name="7", dtype=torch.int64, shape_dims=1),
            ],
            # for constant_as_tensor=True test
            [
                torch.tensor(True),
                torch.tensor(False),
                torch.tensor(1),
                torch.tensor(2.0),
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
            ],
        ],
        # dict
        [
            {"a": True, "b": False, "c": 1, "d": 2.0, "e": "abc", "f": None},
            [],
            {"a": True, "b": False, "c": 1, "d": 2.0, "e": "abc", "f": None},
            # for constant_as_tensor=True test
            [torch.tensor(True), torch.tensor(False), torch.tensor(1), torch.tensor(2.0)],
        ],
        [
            {"a": True, "b": False, "c": 1, "d": 2.0, "e": "abc", "f": None, "g": torch.tensor([1, 2, 3])},
            [torch.tensor([1, 2, 3])],
            {
                "a": True,
                "b": False,
                "c": 1,
                "d": 2.0,
                "e": "abc",
                "f": None,
                "g": _TensorStub(tensor_idx=0, name="g", dtype=torch.int64, shape_dims=1),
            },
            # for constant_as_tensor=True test
            [torch.tensor(True), torch.tensor(False), torch.tensor(1), torch.tensor(2.0), torch.tensor([1, 2, 3])],
        ],
        # list of list
        [
            [[True, False, 1, 2.0, "abc", None]],
            [],
            [[True, False, 1, 2.0, "abc", None]],
            # for constant_as_tensor=True test
            [torch.tensor(True), torch.tensor(False), torch.tensor(1), torch.tensor(2.0)],
        ],
        [
            [[True, False, 1, 2.0, "abc", None, torch.tensor([1, 2, 3])]],
            [torch.tensor([1, 2, 3])],
            [
                [
                    True,
                    False,
                    1,
                    2.0,
                    "abc",
                    None,
                    _TensorStub(tensor_idx=0, name="0_6", dtype=torch.int64, shape_dims=1),
                ]
            ],
            # for constant_as_tensor=True test
            [torch.tensor(True), torch.tensor(False), torch.tensor(1), torch.tensor(2.0), torch.tensor([1, 2, 3])],
        ],
        # list of dict
        [
            [{"a": True, "b": False, "c": 1, "d": 2.0, "e": "abc", "f": None}],
            [],
            [{"a": True, "b": False, "c": 1, "d": 2.0, "e": "abc", "f": None}],
            # for constant_as_tensor=True test
            [torch.tensor(True), torch.tensor(False), torch.tensor(1), torch.tensor(2.0)],
        ],
        [
            [{"a": True, "b": False, "c": 1, "d": 2.0, "e": "abc", "f": None, "g": torch.tensor([1, 2, 3])}],
            [torch.tensor([1, 2, 3])],
            [
                {
                    "a": True,
                    "b": False,
                    "c": 1,
                    "d": 2.0,
                    "e": "abc",
                    "f": None,
                    "g": _TensorStub(tensor_idx=0, name="0_g", dtype=torch.int64, shape_dims=1),
                }
            ],
            # for constant_as_tensor=True test
            [torch.tensor(True), torch.tensor(False), torch.tensor(1), torch.tensor(2.0), torch.tensor([1, 2, 3])],
        ],
        # dict of list
        [
            {"a": [True, False, 1, 2.0, "abc", None]},
            [],
            {"a": [True, False, 1, 2.0, "abc", None]},
            # for constant_as_tensor=True test
            [torch.tensor(True), torch.tensor(False), torch.tensor(1), torch.tensor(2.0)],
        ],
        [
            {"a": [True, False, 1, 2.0, "abc", None, torch.tensor([1, 2, 3])]},
            [torch.tensor([1, 2, 3])],
            {
                "a": [
                    True,
                    False,
                    1,
                    2.0,
                    "abc",
                    None,
                    _TensorStub(tensor_idx=0, name="a_6", dtype=torch.int64, shape_dims=1),
                ]
            },
            # for constant_as_tensor=True test
            [torch.tensor(True), torch.tensor(False), torch.tensor(1), torch.tensor(2.0), torch.tensor([1, 2, 3])],
        ],
        # dict of dict
        [
            {"a": {"b": torch.tensor([1, 2, 3]), "c": torch.tensor([4, 5, 6])}},
            [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
            {
                "a": {
                    "b": _TensorStub(tensor_idx=0, name="a_b", dtype=torch.int64, shape_dims=1),
                    "c": _TensorStub(tensor_idx=1, name="a_c", dtype=torch.int64, shape_dims=1),
                }
            },
            # for constant_as_tensor=True test
            [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
        ],
        # list of mixed types
        [
            [[torch.tensor([[1.3]]), {"a": True}], {"b": torch.tensor([1, 2, 3]), "c": [torch.tensor([4, 5]), 2.0]}],
            [torch.tensor([[1.3]]), torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
            [
                [_TensorStub(tensor_idx=0, name="0_0", dtype=torch.float32, shape_dims=2), {"a": True}],
                {
                    "b": _TensorStub(tensor_idx=1, name="1_b", dtype=torch.int64, shape_dims=1),
                    "c": [_TensorStub(tensor_idx=2, name="1_c_0", dtype=torch.int64, shape_dims=1), 2.0],
                },
            ],
            # for constant_as_tensor=True test
            [
                torch.tensor([[1.3]]),
                torch.tensor(True),
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5]),
                torch.tensor(2.0),
            ],
        ],
    ],
)
@pytest.mark.parametrize(
    "flag",
    [0, 1, 2],
)  # 0: flatten, 1: unflatten, 2: flatten and unflatten
def test_data_flatten_and_unflatten(input_output_map, flag: int):
    raw_data = input_output_map[0]
    flatten_data = input_output_map[1]
    flatten_schema = input_output_map[2]

    def _recursive_compare(real, expected):
        assert type(real) == type(expected)
        if isinstance(real, str):
            assert real == expected
        elif isinstance(real, abc.Sequence):
            assert len(real) == len(expected)
            for i in range(len(real)):
                _recursive_compare(real[i], expected[i])
        elif isinstance(real, abc.Mapping):
            assert len(real.keys()) == len(expected.keys())
            for real_key, real_value in real.items():
                _recursive_compare(real_value, expected[real_key])
        else:
            if isinstance(real, torch.Tensor):
                assert torch.allclose(real, expected)
            else:
                assert real == expected

    if flag == 0:
        out, schema = extract_data_and_schema(raw_data)
        assert all([torch.allclose(o, d) if isinstance(o, torch.Tensor) else o == d for o, d in zip(out, flatten_data)])
        if not isinstance(raw_data, torch.Tensor):
            assert type(schema) == type(raw_data)

        assert str(schema) == str(flatten_schema)

        flatten_data_constant_as_tensor = input_output_map[3]
        out, schema = extract_data_and_schema(raw_data, constant_as_tensor=True, device=torch.device("cpu"))
        if isinstance(
            raw_data,
            (
                type(None),
                str,
            ),
        ):
            assert raw_data == schema
        else:
            assert all(
                [
                    torch.allclose(o, d) if isinstance(o, torch.Tensor) else o == d
                    for o, d in zip(out, flatten_data_constant_as_tensor)
                ]
            )

    elif flag == 1:
        restored_data = unflatten_data_using_schema(flatten_data, flatten_schema)
        _recursive_compare(restored_data, raw_data)
    elif flag == 2:
        out, schema = extract_data_and_schema(raw_data)
        restored_data = unflatten_data_using_schema(out, schema)

        _recursive_compare(restored_data, raw_data)
