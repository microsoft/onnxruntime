# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# model.py

import typing
from abc import abstractmethod

import onnx

import onnxruntime.training.onnxblock._graph_utils as graph_utils
import onnxruntime.training.onnxblock.building_blocks as building_blocks
import onnxruntime.training.onnxblock.model_accessor as accessor


class Model(building_blocks.Block):
    """Builds the forward model based on user's build method."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def build(self, *args, **kwargs):
        """Customize and build the forward graph for this model.

        This method is to be overridden by the user's implementation.
        """
        ...

    def __call__(self, *args, **kwargs):
        """Calls the user's build method and runs validation on top.

        The output onnx model is got by invoking the user's build method.
        """
        # build the user model
        output = self.build(*args, **kwargs)

        # Perform shape inference
        model_with_shapes = onnx.shape_inference.infer_shapes(accessor.global_accessor.model)
        accessor.global_accessor.model.CopyFrom(model_with_shapes)

        # Build the graph outputs
        graph_utils.build_graph_outputs(accessor.global_accessor.model, output)

        return output


class TrainingModel(building_blocks.Block):
    """Builds the training model based on user's build method."""

    def __init__(self):
        super().__init__()
        self._arg_requiring_grad = set()
        self._arg_not_requiring_grad = set()
        self._parameters = None

    @abstractmethod
    def build(self, *args, **kwargs):
        """Customize and build the forward graph for this training model.

        This method is to be overridden by the user's implementation.
        """
        ...

    def requires_grad(self, argument_name: str, value: typing.Optional[bool] = True):
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
            raise RuntimeError("Please build the training model first before trying to retrieve the parameters.")

        return self._parameters

    def __call__(self, *args, **kwargs):
        """Calls the user's build method and builds the gradient graph on top.

        The onnx model contains the user's training model such that:
        1. It contains the gradient graph.
        2. It contains inputs in the order: user inputs, weight parameters,
           gradient inputs.
        3. It contains the outputs in the order: user outputs, gradient outputs.
        4. Before the gradient is built, the eval model is stored in the global accessor.

        Note that the model parameters are moved to be graph inputs.
        """
        # build the user model
        output = self.build(*args, **kwargs)

        # Perform shape inference
        model_with_shapes = onnx.shape_inference.infer_shapes(accessor.global_accessor.model)
        accessor.global_accessor.model.CopyFrom(model_with_shapes)

        # Build the graph outputs
        graph_utils.build_graph_outputs(accessor.global_accessor.model, output)

        # get all the model parameters for the user_model
        # and store them in self._parameters
        self._parameters = graph_utils.get_model_parameters(
            accessor.global_accessor.model, self._arg_not_requiring_grad
        )

        # build the gradient graph
        all_args_requiring_gradient_names = graph_utils.build_gradient_graph(
            accessor.global_accessor,
            self._arg_requiring_grad,
            self._arg_not_requiring_grad,
            output,
        )

        # add gradient accumulation nodes
        graph_utils.build_gradient_accumulation_graph(accessor.global_accessor.model, all_args_requiring_gradient_names)

        return output
