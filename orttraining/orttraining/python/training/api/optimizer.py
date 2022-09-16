# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# optimizer.py

from onnxruntime.capi import _pybind_state as C


class Optimizer:
    """
    Class for running Optimize Step in Training.
    This class is a wrapper of Optimizer Class.
    """

    def __init__(self, train_optimizer_uri, model) -> None:
        """
        Initializes Optimizer with the optimizer onnx and the parameters from the model.
        """
        self._optimizer = C.Optimizer(train_optimizer_uri, model._model)

    def step(self):
        """
        Run Optimizer Step.
        """
        self._optimizer.optimizer_step()
