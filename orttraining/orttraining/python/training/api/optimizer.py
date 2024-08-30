# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from onnxruntime.capi import _pybind_state as C

if TYPE_CHECKING:
    from onnxruntime.training.api.module import Module


class Optimizer:
    """Class that provides methods to update the model parameters based on the computed gradients.

    Args:
        optimizer_uri: The path to the optimizer model.
        model: The module to be trained.
    """

    def __init__(self, optimizer_uri: str | os.PathLike, module: Module):
        self._optimizer = C.Optimizer(
            os.fspath(optimizer_uri), module._state._state, module._device, module._session_options
        )

    def step(self) -> None:
        """Updates the model parameters based on the computed gradients.

        This method updates the model parameters by taking a step in the direction of the computed gradients.
        The optimizer used depends on the optimizer model provided.
        """
        self._optimizer.optimizer_step()

    def set_learning_rate(self, learning_rate: float) -> None:
        """Sets the learning rate for the optimizer.

        Args:
            learning_rate: The learning rate to be set.
        """
        self._optimizer.set_learning_rate(learning_rate)

    def get_learning_rate(self) -> float:
        """Gets the current learning rate of the optimizer.

        Returns:
            The current learning rate.
        """
        return self._optimizer.get_learning_rate()
