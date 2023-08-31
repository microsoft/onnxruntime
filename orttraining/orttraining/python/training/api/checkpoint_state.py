# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import os

import numpy as np

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue


class Parameter:
    """Class that represents a model parameter

    This class represents a model parameter and provides access to its data,
    gradient and other properties. This class is not expected to be instantiated directly.
    Instead, it is returned by the `CheckpointState` object.

    Args:
        parameter: The C.Parameter object that holds the underlying parameter data.
        state: The C.CheckpointState object that holds the underlying session state.
    """

    def __init__(self, parameter: C.Parameter, state: C.CheckpointState):
        self._parameter = parameter
        self._state = state

    @property
    def name(self) -> str:
        """The name of the parameter"""
        return self._parameter.name

    @property
    def data(self) -> np.ndarray:
        """The data of the parameter"""
        return self._parameter.data.numpy()

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """Sets the data of the parameter"""
        self._parameter.copy_from(self._state, OrtValue.ortvalue_from_numpy(value)._ortvalue)

    @property
    def grad(self) -> np.ndarray:
        """The gradient of the parameter"""
        return self._parameter.grad.numpy() if self._parameter.grad.has_value() else None

    @property
    def requires_grad(self) -> bool:
        """Whether or not the parameter requires its gradient to be computed"""
        return self._parameter.requires_grad

    def __repr__(self) -> str:
        """Returns a string representation of the parameter"""
        return f"Parameter(name={self.name}, requires_grad={self.requires_grad})"


class CheckpointState:
    """Class that holds the state of the training session

    This class holds all the state information of the training session such as the model parameters,
    its gradients, the optimizer state and user defined properties.

    User defined properties can be indexed by name from the `CheckpointState` object.

    To create the `CheckpointState`, use the `CheckpointState.load_checkpoint` method.

    Args:
        state: The C.Checkpoint state object that holds the underlying session state.
    """

    def __init__(self, state: C.CheckpointState):
        if not isinstance(state, C.CheckpointState):
            raise TypeError(f"Invalid argument for CheckpointState received {type(state)}")
        self._state = state

    @classmethod
    def load_checkpoint(cls, checkpoint_uri: str | os.PathLike) -> CheckpointState:
        """Loads the checkpoint state from the checkpoint file

        Args:
            checkpoint_uri: The path to the checkpoint file.

        Returns:
            CheckpointState: The checkpoint state object.
        """
        return cls(C.load_checkpoint(os.fspath(checkpoint_uri)))

    @classmethod
    def save_checkpoint(
        cls, state: CheckpointState, checkpoint_uri: str | os.PathLike, include_optimizer_state: bool = False
    ) -> None:
        """Saves the checkpoint state to the checkpoint file

        Args:
            state: The checkpoint state object.
            checkpoint_uri: The path to the checkpoint file.
            include_optimizer_state: If True, the optimizer state is also saved to the checkpoint file.
        """
        C.save_checkpoint(state._state, os.fspath(checkpoint_uri), include_optimizer_state)

    def __getitem__(self, name: str) -> int | float | str | Parameter:
        """Gets the parameter or property associated with the given name

        Searches for the name in the parameters and properties of the checkpoint state.

        Args:
            name: The name of the parameter or property

        Returns:
            The value of the parameter or property
        """

        if self._state.has_parameter(name):
            return Parameter(self._state.get_parameter(name), self._state)
        elif self._state.has_property(name):
            return self._state.get_property(name)
        else:
            raise KeyError(f"Could not find {name} in the checkpoint state.")

    def __setitem__(self, name: str, value: int | float | str | np.ndarray) -> None:
        """Sets the parameter or property value for the given name

        Searches for the name in the parameters and properties of the checkpoint state.
        If the name is found in parameters, the value is updated.
        Else, the value is added or updated in the properties.

        Args:
            name: The name of the parameter or property
            value: The value of the parameter or property
                   Properties only support int, float and str values.
        """
        if self._state.has_parameter(name):
            self._state.copy_parameter_from(name, OrtValue.ortvalue_from_numpy(value)._ortvalue)
        else:
            self._state.add_property(name, value)

    def __contains__(self, name: str) -> bool:
        """Checks if the parameter or property exists in the state

        Tthe name is searched in both parameters and properties.

        Args:
            name: The name of the parameter or property

        Returns:
            True if the name is either a parameter or a property, False otherwise
        """

        return self._state.has_parameter(name) or self._state.has_property(name)
