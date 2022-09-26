# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# checkpoint_state.py

from onnxruntime.capi import _pybind_state as C


class CheckpointState:
    """
    Class for Loading CheckpointState.
    This class is a wrapper of CheckpointState Class.
    """

    def __init__(self, ckpt_uri) -> None:
        """
        Initializes CheckpointState object with the given checkpoint uri.
        The returned object will be used to initialize the Module.
        """
        self._state = C.CheckpointState(ckpt_uri)
