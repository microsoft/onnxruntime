# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# model_accessor.py

from contextlib import contextmanager

import onnx


class ModelAccessor:
    """This class stores the onnx model that is manipulated by the onnx blocks."""

    def __init__(self, model):
        self._model = model
        self._eval_model = None

    @property
    def model(self):
        """ModelAccessor property that gets the modified model."""

        if self._model is None:
            raise RuntimeError(
                "The onnx model was not set. Please use the context manager onnxblock.onnx_model to create the model."
            )
        return self._model

    @property
    def eval_model(self):
        """ModelAccessor property that gets the eval model."""

        if self._eval_model is None:
            raise RuntimeError("The eval onnx model was not set.")
        return self._eval_model

    @eval_model.setter
    def eval_model(self, value):
        """ModelAccessor property that sets the eval model."""
        self._eval_model = value


# This variable resides in the global namespace.
# Different methods can access this global model and manipulate it.
# Its construction and destruction is managed by the onnx_model contextmanager
global_accessor = None


@contextmanager
def onnx_model(model=None):
    """Context manager that is the entry point to graph manipulations on model.

    Manages the construction and destruction of the global model.
    """
    global global_accessor
    if global_accessor is not None:
        raise RuntimeError("Base onnx model already exists. Cannot create multiple ModelAccessors.")

    # If the user did not provide a model, then assume that they want to build from scratch.
    # It is the duty of the caller to fill the model however they deem fit.
    if model is None:
        model = onnx.ModelProto()

    global_accessor = ModelAccessor(model)
    try:
        yield global_accessor
    finally:
        global_accessor = None
