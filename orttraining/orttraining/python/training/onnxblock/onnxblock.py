# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from abc import abstractmethod
from typing import List, Tuple

import onnx

import onnxruntime.training.onnxblock._graph_utils as _graph_utils
import onnxruntime.training.onnxblock._training_graph_utils as _training_graph_utils
import onnxruntime.training.onnxblock.blocks as blocks
import onnxruntime.training.onnxblock.model_accessor as accessor


class ForwardBlock(blocks.Block):
    """Base class for all blocks that require forward model to be automatically built.

    Blocks wanting to build a forward model by stacking blocks on top of the existing model
    must subclass this class. The subclass's implementation of the build method must return the
    name of the graph output. This block will automatically register the output as a graph output
    and build the model.

    Example:

    >>> class MyForwardBlock(ForwardBlock):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         self.loss = onnxblock.loss.CrossEntropyLoss()
    >>>
    >>>     def build(self, loss_input_name: str):
    >>>         # Add a cross entropy loss on top of the output so far (loss_input_name)
    >>>         return self.loss(loss_input_name)

    The above example will automatically build the forward graph that is composed of the existing model
    and the cross entropy loss function stacked on top of it.
    """

    def __init__(self):
        super().__init__()
        self._model = None

    @abstractmethod
    def build(self, *args, **kwargs):
        """Customize the forward graph for this model by stacking up blocks on top of the inputs to this function.

        This method should be overridden by the subclass. The output of this method should
        be the name of the graph output.
        """

    def to_model_proto(self):
        """Returns the forward model.

        Returns:
            model (onnx.ModelProto): The forward model.

        Raises:
            RuntimeError: If the build method has not been invoked (i.e. the forward model has not been built yet).
        """
        if self._model is None:
            raise RuntimeError("Please build the forward models first before trying to retrieve it.")
        return self._model

    def __call__(self, *args, **kwargs):
        """Calls the subclass's build method and builds the forward model."""

        self.base = accessor._GLOBAL_ACCESSOR.model

        logging.debug("Building forward block %s", self.__class__.__name__)

        output = self.build(*args, **kwargs)

        self._model = onnx.shape_inference.infer_shapes(accessor._GLOBAL_ACCESSOR.model)

        _graph_utils.register_graph_outputs(self._model, output)

        accessor._GLOBAL_ACCESSOR.model.CopyFrom(self._model)

        return output


class TrainingBlock(blocks.Block):
    """Base class for all blocks that require gradient model to be automatically built.

    Blocks that require the gradient graph to be computed based on the output of the block
    must subclass this class. The subclass's implementation of the build method must return
    the name of the output from where backpropagation must begin (typically the name of the
    output from the loss function).

    Example:

    >>> class MyTrainingBlock(TrainingBlock):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         self.loss = onnxblock.loss.CrossEntropyLoss()
    >>>
    >>>     def build(self, loss_input_name: str):
    >>>         # Add a cross entropy loss on top of the output so far (loss_input_name)
    >>>         return self.loss(loss_input_name)

    The above example will automatically build the gradient graph for the entire model
    starting from the output of the loss function.
    """

    def __init__(self):
        super().__init__()
        self._requires_grad = set()
        self._frozen_params = set()
        self._parameters = None
        self._training_model = None
        self._eval_model = None

    @abstractmethod
    def build(self, *args, **kwargs):
        """Customize the forward graph for this model by stacking up blocks on top of the inputs to this function.

        This method should be overridden by the subclass. The output of this method should
        be the name of the output from where backpropagation must begin (typically the name
        of the output from the loss function).
        """

    def requires_grad(self, argument_name: str, value: bool = True):
        """Specify whether the argument requires gradient or not.

        The auto-diff will compute the gradient graph for only the arguments that
        require gradient. By default, none of the arguments require gradient.
        The user must explicitly specify which arguments require gradient.

        Args:
            argument_name (str): The name of the argument that require/does not require gradient.
            value (bool): True if the argument requires gradient, False otherwise.
        """
        if value is True:
            self._requires_grad.add(argument_name)
            if argument_name in self._frozen_params:
                self._frozen_params.remove(argument_name)
        else:
            if argument_name in self._requires_grad:
                self._requires_grad.remove(argument_name)
            self._frozen_params.add(argument_name)

    def parameters(self) -> Tuple[List[onnx.TensorProto], List[onnx.TensorProto]]:
        """Trainable as well as non-trainable (frozen) parameters of the model.

        Model parameters that are extracted while building the training model
        are returned by this method.

        Note that the parameters are not known before the training model is
        built. As a result, if this method is invoked before the training model
        is built, an exception will be raised.

        Returns:
            trainable_params (list of onnx.TensorProto): The trainable parameters of the model.
            frozen_params (list of onnx.TensorProto): The non-trainable parameters of the model.

        Raises:
            RuntimeError: If the build method has not been invoked (i.e. the training model has not been built yet).
        """
        if self._parameters is None:
            raise RuntimeError("Please build the training model first before trying to retrieve the parameters.")

        return self._parameters

    def to_model_proto(self) -> Tuple[onnx.ModelProto, onnx.ModelProto]:
        """Returns the training and eval models.

        Once the gradient graph is built, the training and eval models can be retrieved
        by invoking this method.

        Returns:
            training_model (onnx.ModelProto): The training model.
            eval_model (onnx.ModelProto): The eval model.

        Raises:
            RuntimeError: If the build method has not been invoked (i.e. the training model has not been built yet).
        """
        if self._training_model is None or self._eval_model is None:
            raise RuntimeError("Please build the training and eval models first before trying to retrieve them.")
        return self._training_model, self._eval_model

    def __call__(self, *args, **kwargs):
        """Calls the subclass's build method and builds the gradient graph on top."""

        self.base = accessor._GLOBAL_ACCESSOR.model

        logging.debug("Building training block %s", self.__class__.__name__)

        output = self.build(*args, **kwargs)

        model = onnx.shape_inference.infer_shapes(accessor._GLOBAL_ACCESSOR.model)

        _graph_utils.register_graph_outputs(model, output)

        logging.debug("Building gradient graph for training block %s", self.__class__.__name__)

        self._parameters = _training_graph_utils.get_model_parameters(model, self._requires_grad, self._frozen_params)

        # Build the gradient graph. The gradient graph building is composed of the following steps:
        #   - Move all model parameters to model inputs.
        #   - Run orttraining graph transformers on the model.
        #   - Add the gradient graph to the optimized model.
        # The order of model inputs after gradient graph building is: user inputs, model parameters as inputs
        # The order of the model outputs is: user outputs, model parameter gradients (in the order of parameter inputs)
        self._training_model, self._eval_model = _training_graph_utils.build_gradient_graph(
            model, self._requires_grad, self._frozen_params, output, accessor._GLOBAL_CUSTOM_OP_LIBRARY
        )

        logging.debug("Adding gradient accumulation nodes for training block %s", self.__class__.__name__)

        _training_graph_utils.build_gradient_accumulation_graph(self._training_model, self._requires_grad)

        accessor._GLOBAL_ACCESSOR.model.CopyFrom(self._training_model)

        return output
