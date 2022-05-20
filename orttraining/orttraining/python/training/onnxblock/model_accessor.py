# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# global_model.py

from contextlib import contextmanager


class ModelAccessor:
    """This class stores the onnx model that is manipulated by the onnx blocks."""

    def __init__(self, model):
        self.model = model
        self.eval_model = None


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
    global_accessor = ModelAccessor(model)
    try:
        yield global_accessor
    finally:
        global_accessor = None
