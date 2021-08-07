# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import h5py
from collections.abc import Mapping
import pickle

def _dfs_save(group, save_obj):
    """Recursively go over each level in the save_obj dictionary and save values to a hdf5 group"""

    for key, value in save_obj.items():
        if isinstance(value, Mapping):
            subgroup = group.create_group(key)
            _dfs_save(subgroup, value)
        else:
            group[key] = value

def save(save_obj: dict, path):
    """Persists the input dictionary to a file specified by path.

    Saves an hdf5 representation of the save_obj dictionary to a file or a file-like object specified by path.
    Values are saved in a format supported by h5py. For example, a PyTorch tensor is saved and loaded as a
    numpy object. So, user types may be converted from their original types to numpy equivalent types.

    Args:
        save_obj: dictionary that needs to be saved.
            save_obj should consist of types supported by hdf5 file format.
            if hdf5 does not recognize a type, an exception is raised.
            if save_obj is not a dictionary, a ValueError is raised.
        path: string representation to a file path or a python file-like object.
            if file already exists at path, an exception is raised.
    """
    if not isinstance(save_obj, Mapping):
        raise ValueError("Object to be saved must be a dictionary")

    with h5py.File(path, 'w-') as f:
        _dfs_save(f, save_obj)

def _dfs_load(group, load_obj):
    """Recursively go over each level in the hdf5 group and load the values into the given dictionary"""

    for key in group:
        if isinstance(group[key], h5py.Group):
            load_obj[key] = {}
            _dfs_load(group[key], load_obj[key])
        else:
            load_obj[key] = group[key][()]

def load(path, key=None):
    """Loads the data stored in the binary file specified at the given path into a dictionary and returns it.

    Loads the data from an hdf5 file specified at the given path into a python dictionary.
    Loaded dictionary contains numpy equivalents of python data types. For example:
        PyTorch tensor -> saved as a numpy array and loaded as a numpy array.
        bool -> saved as a numpy bool and loaded as a numpy bool
    If a '/' separated key is provided, the value at that hierarchical level in the hdf5 group is returned.

    Args:
        path: string representation to a file path or a python file-like object.
            if file does not already exist at path, an exception is raised.
        key: '/' separated representation of the hierarchy level value that needs to be returned/
            for example, if the saved binary file has structure {a: {b: x, c:y}} and the user would like
            to query the value for c, the key provided should be 'a/c'.
            the default value of None for key implies that the entire hdf5 file structure needs to be loaded into a dictionary and returned.

    Returns:
        a dictionary loaded from the specified binary hdf5 file.
    """
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path} is not an hdf5 file or a python file-like object.")

    load_obj = {}
    with h5py.File(path, 'r') as f:
        if key:
            f = f[key]
        if isinstance(f, h5py.Dataset):
            return f[()]

        _dfs_load(f, load_obj)

    return load_obj

def to_serialized_hex(user_dict):
    """Serialize the user_dict and convert the serialized bytes to a hex string and return"""

    return pickle.dumps(user_dict).hex()

def from_serialized_hex(serialized_hex):
    """Convert serialized_hex to bytes and deserialize it and return"""

    # serialized_hex can be either a regular string or a byte string.
    # if it is a byte string, convert to regular string using decode()
    # if it is a regular string, do nothing to it
    try:
        serialized_hex = serialized_hex.decode()
    except AttributeError:
        pass
    return pickle.loads(bytes.fromhex(serialized_hex))
