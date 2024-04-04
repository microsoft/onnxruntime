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


class Parameters:
    """Class that holds all the model parameters

    This class holds all the model parameters and provides access to them.
    This class is not expected to be instantiated directly. Instead, it is returned by the
    `CheckpointState`'s parameters attribute.
    This class behaves like a dictionary and provides access to the parameters by name.

    Args:
        state: The C.CheckpointState object that holds the underlying session state.
    """

    def __init__(self, state: C.CheckpointState):
        self._state = state

    def __getitem__(self, name: str) -> Parameter:
        """Gets the parameter associated with the given name

        Searches for the name in the parameters of the checkpoint state.

        Args:
            name: The name of the parameter

        Returns:
            The value of the parameter

        Raises:
            KeyError: If the parameter is not found
        """

        if name not in self:
            raise KeyError(f"Parameter {name} not found.")

        return Parameter(self._state.get_parameter(name), self._state)

    def __setitem__(self, name: str, value: np.ndarray) -> None:
        """Sets the parameter value for the given name

        Searches for the name in the parameters of the checkpoint state.
        If the name is found in parameters, the value is updated.

        Args:
            name: The name of the parameter
            value: The value of the parameter as a numpy array

        Raises:
            KeyError: If the parameter is not found
        """
        if name not in self:
            raise KeyError(f"Parameter {name} not found.")

        self._state.copy_parameter_from(name, OrtValue.ortvalue_from_numpy(value)._ortvalue)

    def __contains__(self, name: str) -> bool:
        """Checks if the parameter exists in the state

        Args:
            name: The name of the parameter

        Returns:
            True if the name is a parameter False otherwise
        """

        return self._state.has_parameter(name)

    def __iter__(self):
        """Returns an iterator over the properties"""
        for parameter_name in self._state.parameter_names():
            yield parameter_name, Parameter(self._state.get_parameter(parameter_name), self._state)

    def __repr__(self) -> str:
        """Returns a string representation of the parameters"""
        return self._state.parameter_names()

    def __len__(self) -> int:
        """Returns the number of parameters"""
        return len(self._state.parameter_names())


class Properties:
    def __init__(self, state: C.CheckpointState):
        self._state = state

    def __getitem__(self, name: str) -> int | float | str:
        """Gets the property associated with the given name

        Searches for the name in the properties of the checkpoint state.

        Args:
            name: The name of the property

        Returns:
            The value of the property

        Raises:
            KeyError: If the property is not found
        """

        if name not in self:
            raise KeyError(f"Property {name} not found.")

        return self._state.get_property(name)

    def __setitem__(self, name: str, value: int | float | str) -> None:
        """Sets the property value for the given name

        Searches for the name in the properties of the checkpoint state.
        The value is added or updated in the properties.

        Args:
            name: The name of the property
            value: The value of the property
                   Properties only support int, float and str values.
        """
        self._state.add_property(name, value)

    def __contains__(self, name: str) -> bool:
        """Checks if the property exists in the state

        Args:
            name: The name of the property

        Returns:
            True if the name is a property, False otherwise
        """

        return self._state.has_property(name)

    def __iter__(self):
        """Returns an iterator over the properties"""
        for property_name in self._state.property_names():
            yield property_name, self._state.get_property(property_name)

    def __repr__(self) -> str:
        """Returns a string representation of the properties"""
        return self._state.property_names()

    def __len__(self) -> int:
        """Returns the number of properties"""
        return len(self._state.property_names())


class CheckpointState:
    """Class that holds the state of the training session

    This class holds all the state information of the training session such as the model parameters,
    its gradients, the optimizer state and user defined properties.

    To create the `CheckpointState`, use the `CheckpointState.load_checkpoint` method.

    Args:
        state: The C.Checkpoint state object that holds the underlying session state.
    """

    def __init__(self, state: C.CheckpointState):
        if not isinstance(state, C.CheckpointState):
            raise TypeError(f"Invalid argument for CheckpointState received {type(state)}")
        self._state = state
        self._parameters = Parameters(self._state)
        self._properties = Properties(self._state)

    @classmethod
    def load_checkpoint(cls, checkpoint_uri: str | os.PathLike) -> CheckpointState:
        """Loads the checkpoint state from the checkpoint file

        The checkpoint file can either be the complete checkpoint or the nominal checkpoint.

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

    @property
    def parameters(self) -> Parameters:
        """Returns the model parameters from the checkpoint state"""
        return self._parameters

    @property
    def properties(self) -> Properties:
        """Returns the properties from the checkpoint state"""
        return self._properties
