# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_checkpoint_storage.py

import os
import pickle
import shutil

import numpy as np
import pytest
import torch

from onnxruntime.training import _checkpoint_storage

# Helper functions


def _equals(a, b):
    """Checks recursively if two dictionaries are equal"""
    if isinstance(a, dict):
        return all(not (key not in b or not _equals(a[key], b[key])) for key in a)
    else:
        if isinstance(a, bytes):
            a = a.decode()
        if isinstance(b, bytes):
            b = b.decode()
        are_equal = a == b
        return are_equal if isinstance(are_equal, bool) else are_equal.all()

    return False


def _numpy_types(obj_value):
    """Return a bool indicating whether or not the input obj_value is a numpy type object

    Recursively checks if the obj_value (could be a dictionary) is a numpy type object.
    Exceptions are str and bytes.

    Returns true if object is numpy type, str, or bytes
    False if any other type
    """
    if not isinstance(obj_value, dict):
        return isinstance(obj_value, (str, bytes)) or type(obj_value).__module__ == np.__name__

    return all(_numpy_types(value) for _, value in obj_value.items())


def _get_dict(separated_key):
    """Create dummy dictionary with different datatypes

    Returns the tuple of the entire dummy dictionary created, key argument as a dictionary for _checkpoint_storage.load
    function and the value for that key in the original dictionary

    For example the complete dictionary is represented by:
    {
        'int1':1,
        'int2': 2,
        'int_list': [1,2,3,5,6],
        'dict1': {
            'np_array': np.arange(100),
            'dict2': {'int3': 3, 'int4': 4},
            'str1': "onnxruntime"
        },
        'bool1': bool(True),
        'int5': 5,
        'float1': 2.345,
        'np_array_float': np.array([1.234, 2.345, 3.456]),
        'np_array_float_3_dim': np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
    }

    if the input key is ['dict1', 'str1'], then the key argument returned is 'dict1/str1'
    and the value corresponding to that is "onnxruntime"

    so, for the above example, the returned tuple is:
    (original_dict, {'key': 'dict1/str1', "onnxruntime")
    """
    test_dict = {
        "int1": 1,
        "int2": 2,
        "int_list": [1, 2, 3, 5, 6],
        "dict1": {"np_array": np.arange(100), "dict2": {"int3": 3, "int4": 4}, "str1": "onnxruntime"},
        "bool1": True,
        "int5": 5,
        "float1": 2.345,
        "np_array_float": np.array([1.234, 2.345, 3.456]),
        "np_array_float_3_dim": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    }
    key = ""
    expected_val = test_dict
    for single_key in separated_key:
        key += single_key + "/"
        expected_val = expected_val[single_key]
    return test_dict, {"key": key} if len(separated_key) > 0 else dict(), expected_val


class _CustomClass:
    """Custom object that encpsulates dummy values for loss, epoch and train_step"""

    def __init__(self):
        self._loss = 1.23
        self._epoch = 12000
        self._train_step = 25

    def __eq__(self, other):
        if isinstance(other, _CustomClass):
            return self._loss == other._loss and self._epoch == other._epoch and self._train_step == other._train_step


# Test fixtures


@pytest.yield_fixture(scope="function")
def checkpoint_storage_test_setup():
    checkpoint_dir = os.path.abspath("checkpoint_dir/")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    pytest.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.ortcp")
    yield "checkpoint_storage_test_setup"
    shutil.rmtree(checkpoint_dir)


@pytest.yield_fixture(scope="function")
def checkpoint_storage_test_parameterized_setup(request, checkpoint_storage_test_setup):
    yield request.param


# Tests


@pytest.mark.parametrize(
    "checkpoint_storage_test_parameterized_setup",
    [
        _get_dict([]),
        _get_dict(["int1"]),
        _get_dict(["dict1"]),
        _get_dict(["dict1", "dict2"]),
        _get_dict(["dict1", "dict2", "int4"]),
        _get_dict(["dict1", "str1"]),
        _get_dict(["bool1"]),
        _get_dict(["float1"]),
        _get_dict(["np_array_float"]),
    ],
    indirect=True,
)
def test_checkpoint_storage_saved_dict_matches_loaded(checkpoint_storage_test_parameterized_setup):
    to_save = checkpoint_storage_test_parameterized_setup[0]
    key_arg = checkpoint_storage_test_parameterized_setup[1]
    expected = checkpoint_storage_test_parameterized_setup[2]
    _checkpoint_storage.save(to_save, pytest.checkpoint_path)
    loaded = _checkpoint_storage.load(pytest.checkpoint_path, **key_arg)
    assert _equals(loaded, expected)
    assert _numpy_types(loaded)


@pytest.mark.parametrize(
    "checkpoint_storage_test_parameterized_setup",
    [{"int_set": {1, 2, 3, 4, 5}}, {"str_set": {"one", "two"}}, [1, 2, 3], 2.352],
    indirect=True,
)
def test_checkpoint_storage_saving_non_supported_types_fails(checkpoint_storage_test_parameterized_setup):
    to_save = checkpoint_storage_test_parameterized_setup
    with pytest.raises(Exception):  # noqa: B017
        _checkpoint_storage.save(to_save, pytest.checkpoint_path)


@pytest.mark.parametrize(
    "checkpoint_storage_test_parameterized_setup",
    [
        ({"int64_tensor": torch.tensor(np.arange(100))}, "int64_tensor", torch.int64, np.int64),
        ({"int32_tensor": torch.tensor(np.arange(100), dtype=torch.int32)}, "int32_tensor", torch.int32, np.int32),
        ({"int16_tensor": torch.tensor(np.arange(100), dtype=torch.int16)}, "int16_tensor", torch.int16, np.int16),
        ({"int8_tensor": torch.tensor(np.arange(100), dtype=torch.int8)}, "int8_tensor", torch.int8, np.int8),
        ({"float64_tensor": torch.tensor(np.array([1.0, 2.0]))}, "float64_tensor", torch.float64, np.float64),
        (
            {"float32_tensor": torch.tensor(np.array([1.0, 2.0]), dtype=torch.float32)},
            "float32_tensor",
            torch.float32,
            np.float32,
        ),
        (
            {"float16_tensor": torch.tensor(np.array([1.0, 2.0]), dtype=torch.float16)},
            "float16_tensor",
            torch.float16,
            np.float16,
        ),
    ],
    indirect=True,
)
def test_checkpoint_storage_saving_tensor_datatype(checkpoint_storage_test_parameterized_setup):
    tensor_dict = checkpoint_storage_test_parameterized_setup[0]
    tensor_name = checkpoint_storage_test_parameterized_setup[1]
    tensor_dtype = checkpoint_storage_test_parameterized_setup[2]
    np_dtype = checkpoint_storage_test_parameterized_setup[3]

    _checkpoint_storage.save(tensor_dict, pytest.checkpoint_path)

    loaded = _checkpoint_storage.load(pytest.checkpoint_path)
    assert isinstance(loaded[tensor_name], np.ndarray)
    assert tensor_dict[tensor_name].dtype == tensor_dtype
    assert loaded[tensor_name].dtype == np_dtype
    assert (tensor_dict[tensor_name].numpy() == loaded[tensor_name]).all()


@pytest.mark.parametrize(
    "checkpoint_storage_test_parameterized_setup",
    [
        ({"two_dim": torch.ones([2, 4], dtype=torch.float64)}, "two_dim"),
        ({"three_dim": torch.ones([2, 4, 6], dtype=torch.float64)}, "three_dim"),
        ({"four_dim": torch.ones([2, 4, 6, 8], dtype=torch.float64)}, "four_dim"),
    ],
    indirect=True,
)
def test_checkpoint_storage_saving_multiple_dimension_tensors(checkpoint_storage_test_parameterized_setup):
    tensor_dict = checkpoint_storage_test_parameterized_setup[0]
    tensor_name = checkpoint_storage_test_parameterized_setup[1]

    _checkpoint_storage.save(tensor_dict, pytest.checkpoint_path)

    loaded = _checkpoint_storage.load(pytest.checkpoint_path)
    assert isinstance(loaded[tensor_name], np.ndarray)
    assert (tensor_dict[tensor_name].numpy() == loaded[tensor_name]).all()


@pytest.mark.parametrize(
    "checkpoint_storage_test_parameterized_setup", [{}, {"a": {}}, {"a": {"b": {}}}], indirect=True
)
def test_checkpoint_storage_saving_and_loading_empty_dictionaries_succeeds(checkpoint_storage_test_parameterized_setup):
    saved = checkpoint_storage_test_parameterized_setup
    _checkpoint_storage.save(saved, pytest.checkpoint_path)

    loaded = _checkpoint_storage.load(pytest.checkpoint_path)
    assert _equals(saved, loaded)


def test_checkpoint_storage_load_file_that_does_not_exist_fails(checkpoint_storage_test_setup):
    with pytest.raises(Exception):  # noqa: B017
        _checkpoint_storage.load(pytest.checkpoint_path)


def test_checkpoint_storage_for_custom_user_dict_succeeds(checkpoint_storage_test_setup):
    custom_class = _CustomClass()
    user_dict = {"tensor1": torch.tensor(np.arange(100), dtype=torch.float32), "custom_class": custom_class}

    pickled_bytes = pickle.dumps(user_dict).hex()
    to_save = {"a": torch.tensor(np.array([1.0, 2.0]), dtype=torch.float32), "user_dict": pickled_bytes}
    _checkpoint_storage.save(to_save, pytest.checkpoint_path)

    loaded_dict = _checkpoint_storage.load(pytest.checkpoint_path)
    assert (loaded_dict["a"] == to_save["a"].numpy()).all()
    try:  # noqa: SIM105
        loaded_dict["user_dict"] = loaded_dict["user_dict"].decode()
    except AttributeError:
        pass
    loaded_obj = pickle.loads(bytes.fromhex(loaded_dict["user_dict"]))

    assert torch.all(loaded_obj["tensor1"].eq(user_dict["tensor1"]))
    assert loaded_obj["custom_class"] == custom_class
