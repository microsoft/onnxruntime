# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations
from typing import Union
import os
from onnxruntime.capi import _pybind_state as C


class CheckpointState:
    """Class that holds the state of the training session state

    Args:
        state (CheckpointState): The C.Checkpoint state object that holds the underlying session state.
    """

    def __init__(self, state: C.CheckpointState):
        if not isinstance(state, C.CheckpointState):
            raise TypeError(f"Invalid argument for CheckpointState received {type(state)}")
        self._state = state

    @staticmethod
    def load_checkpoint(checkpoint_uri: str | os.PathLike) -> CheckpointState:
        """Loads the checkpoint state from the checkpoint file

        Args:
            checkpoint_uri: The path to the checkpoint file.

        Returns:
            CheckpointState: The checkpoint state object.
        """
        return CheckpointState(C.load_checkpoint(str(checkpoint_uri)))

    @staticmethod
    def save_checkpoint(state: CheckpointState, checkpoint_uri: str | os.PathLike) -> None:
        """Saves the checkpoint state to the checkpoint file

        Args:
            state: The checkpoint state object.
            checkpoint_uri: The path to the checkpoint file.
        """
        C.save_checkpoint(state._state, str(checkpoint_uri))

    def __getitem__(self, name: str) -> Union[int, float, str]:
        """Gets the property associated with the given name

        Args:
            name: The name of the property

        Returns:
            The value of the property
        """
        return self._state.get_property(name)

    def __setitem__(self, name: str, value: Union[int, float, str]) -> None:
        """Sets the property value for the given name

        Args:
            name: The name of the property
            value: The value of the property
        """
        self._state.add_property(name, value)

    def __contains__(self, name: str) -> bool:
        """Checks if the property exists in the state

        Args:
            name: The name of the property

        Returns:
            True if the property exists, False otherwise
        """
        return self._state.has_property(name)
