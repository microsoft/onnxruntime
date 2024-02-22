# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import os
from contextlib import contextmanager

import onnx


class ModelAccessor:
    """This class stores the onnx model that is manipulated by the onnx blocks.

    Attributes:
        model: The onnx model that is manipulated by the onnx blocks.
    """

    def __init__(self, model: onnx.ModelProto):
        self._model = model

    @property
    def model(self) -> onnx.ModelProto:
        """ModelAccessor property that gets the modified model."""

        if self._model is None:
            raise RuntimeError(
                "The onnx model was not set. Please use the context manager onnxblock.onnx_model to create the model."
            )
        return self._model


# These variable resides in the global namespace.
# Different methods can access this global model and manipulate it.
# Its construction and destruction is managed by the base and empty_base contextmanagers
_GLOBAL_ACCESSOR = None
_GLOBAL_CUSTOM_OP_LIBRARY = None


@contextmanager
def base(model: onnx.ModelProto):
    """Registers the base model to be manipulated by the onnx blocks.

    Example:
    >>> with onnxblock.base(model) as model_handle:
    >>>     # manipulate the model using blocks
    >>>     ...
    >>>     # get the modified model
    >>>     model = model_handle.model

    In this example, base will register the given input model to be manipulated by the onnx blocks.

    Args:
        model: The base model to be manipulated by the onnx blocks.

    Returns:
        ModelAccessor: The model accessor that contains the modified model.
    """
    global _GLOBAL_ACCESSOR  # pylint: disable=global-statement  # noqa: PLW0603
    if _GLOBAL_ACCESSOR is not None:
        raise RuntimeError("Base onnx model already exists. Cannot create multiple ModelAccessors.")

    model_clone = copy.deepcopy(model)

    if model_clone is None:
        raise RuntimeError(
            "Base onnx model cannot be None. Please use onnxblock.empty_base if you would like to build a "
            "model from scratch."
        )

    _GLOBAL_ACCESSOR = ModelAccessor(model_clone)
    try:
        yield _GLOBAL_ACCESSOR
    finally:
        _GLOBAL_ACCESSOR = None


@contextmanager
def empty_base(opset_version: int | None = None):
    """Registers an empty base model to be manipulated by the onnx blocks.

    Example:
    >>> with onnxblock.empty_base() as model_handle:
    >>>     # manipulate the model using blocks
    >>>     ...
    >>>     # get the modified model
    >>>     model = model_handle.model

    In this example, empty_base will register a new ModelProto as the base model. Blocks
    will manipulate this model. The user can then retrieve the modified model from the
    model_handle.

    Args:
        opset_version: The opset version to use for the model. Defaults to onnx.defs.onnx_opset_version()

    Returns:
        ModelAccessor: The model accessor that contains the modified model.
    """
    global _GLOBAL_ACCESSOR  # pylint: disable=global-statement  # noqa: PLW0603
    if _GLOBAL_ACCESSOR is not None:
        raise RuntimeError("Base onnx model already exists. Cannot create multiple ModelAccessors.")

    model = onnx.ModelProto()
    model.ir_version = onnx.IR_VERSION
    model.producer_name = "orttraining"
    model.graph.name = "graph"
    model.opset_import.extend(
        (
            onnx.helper.make_opsetid("com.microsoft", 1),
            onnx.helper.make_opsetid("", opset_version or onnx.defs.onnx_opset_version()),
        )
    )

    _GLOBAL_ACCESSOR = ModelAccessor(model)
    try:
        yield _GLOBAL_ACCESSOR
    finally:
        _GLOBAL_ACCESSOR = None


@contextmanager
def custom_op_library(custom_op_library_path: os.PathLike):
    """Registers the custom op library to be used by the onnx blocks.

    Example:
    >>> with onnxblock.custom_op_library(custom_op_library_path):
    >>>     # manipulate the model using blocks
    >>>     ...

    In this example, custom_op_library will register the given input custom op library path to be used
    during the model manipulation (gradient graph building and optimization).

    Args:
        custom_op_library_path: The path to the custom op library.

    Returns:
        ModelAccessor: The model accessor that contains the modified model.
    """
    global _GLOBAL_CUSTOM_OP_LIBRARY  # pylint: disable=global-statement  # noqa: PLW0603
    if _GLOBAL_CUSTOM_OP_LIBRARY is not None:
        raise RuntimeError("CustomOp library already set. Cannot set multiple custom op libraries.")

    if not os.path.exists(custom_op_library_path):
        raise RuntimeError(f"Custom op library path {custom_op_library_path} does not exist.")

    _GLOBAL_CUSTOM_OP_LIBRARY = copy.copy(custom_op_library_path)
    try:
        yield _GLOBAL_CUSTOM_OP_LIBRARY
    finally:
        _GLOBAL_CUSTOM_OP_LIBRARY = None
