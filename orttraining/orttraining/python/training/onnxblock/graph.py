# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# graph.py

from abc import ABC, abstractmethod
import onnx
from ._graph_utils import (
    build_gradient_model,
    build_gradient_accumulation_model,
    get_model_parameters,
)


class Graph(ABC):
    """Builds the forward model based on user's build method."""

    def __init__(self):
        ...

    @abstractmethod
    def build(self, *args, **kwargs):
        """Customize and build the forward graph for this model.

        This method is to be overriden by the user's implementation.
        """
        ...

    def __call__(self, *args, **kwargs):
        """Calls the user's build method and runs validation on top.

        The output onnx model is got by invoking the user's build method.
        """
        # build the user model
        user_model = self.build(*args, **kwargs)

        # validate and check the model
        onnx.checker.check_model(user_model, True)

        return user_model


class TrainingGraph(Graph):
    """Builds the training model based on user's build method."""

    def __init__(self):
        super(TrainingGraph, self).__init__()
        self._arg_requiring_grad = set()
        self._arg_not_requiring_grad = set()
        self._parameters = None

    @abstractmethod
    def build(self, *args, **kwargs):
        """Customize and build the forward graph for this training model.

        This method is to be overriden by the user's implementation.
        """
        ...

    def requires_grad(self, argument_name, value=True):
        """Control whether the given graph input/parameter requires gradient."""
        if value is True:
            if argument_name in self._arg_not_requiring_grad:
                self._arg_not_requiring_grad.remove(argument_name)
            self._arg_requiring_grad.add(argument_name)
        else:
            if argument_name in self._arg_requiring_grad:
                self._arg_requiring_grad.remove(argument_name)
            self._arg_not_requiring_grad.add(argument_name)

    def parameters(self):
        """Returns trainable and non trainable parameters.

        Model parameters that are extracted while building the training model
        are returned by this method.

        Note that the parameters are not known before the training model is
        built. As a result, if this method is invoked before the training model
        is built, an exception will be raised.
        """
        if self._parameters is None:
            raise RuntimeError(
                "Please build the training graph first before trying to "
                "retrieve the parameters."
            )

        return self._parameters

    def __call__(self, *args, **kwargs):
        """Calls the user's build method and builds the gradient graph on top.

        The output onnx model contains the user's training model such that:
        1. It contains the gradient graph.
        2. It contains inputs in the order: user inputs, weight parameters,
           gradient inputs.
        3. It contains the outputs in the order: user outputs, gradient outputs.

        Note that the model parameters are moved to be graph inputs.
        """
        # build the user model
        user_model = self.build(*args, **kwargs)

        # get all the model parameters for the user_model
        # and store them in self._parameters
        self._parameters = get_model_parameters(
            user_model, self._arg_not_requiring_grad
        )

        # build the gradient graph
        grad_model = build_gradient_model(
            user_model, self._arg_requiring_grad, self._arg_not_requiring_grad
        )

        # add gradient accumulation nodes
        grad_model = build_gradient_accumulation_model(grad_model)

        # validate and check the model
        onnx.checker.check_model(grad_model, True)

        return grad_model
