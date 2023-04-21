# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import os

import numpy as np

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue, get_ort_device_type
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector
from onnxruntime.training.api.checkpoint_state import CheckpointState


class Module:
    """Trainer class that provides training and evaluation methods for ONNX models.

    Before instantiating the Module class, it is expected that the training artifacts have been
    generated using the `onnxruntime.training.artifacts.generate_artifacts` utility.

    The training artifacts include:
        - The training model
        - The evaluation model (optional)
        - The optimizer model (optional)
        - The checkpoint file

    Args:
        train_model_uri: The path to the training model.
        state: The checkpoint state object.
        eval_model_uri: The path to the evaluation model.
        device: The device to run the model on. Default is "cpu".
    """

    training: bool

    def __init__(
        self,
        train_model_uri: os.PathLike,
        state: CheckpointState,
        eval_model_uri: os.PathLike | None = None,
        device: str = "cpu",
    ) -> None:
        self.training = True
        options = device.split(":")
        self._device_type = options[0]
        device_id = 0 if len(options) < 2 else int(options[1])

        self._device = C.OrtDevice(
            get_ort_device_type(self._device_type, device_id),
            C.OrtDevice.default_memory(),
            device_id,
        )
        self._model = C.Module(
            os.fspath(train_model_uri),
            state._state,
            os.fspath(eval_model_uri) if eval_model_uri is not None else None,
            self._device,
        )
        self._state = state

    def __call__(self, *user_inputs) -> tuple[np.ndarray] | np.ndarray:
        """Invokes either the training or the evaluation step of the model.

        Args:
            user_inputs: The inputs to the model.
        Returns:
            fetches : The outputs of the model.
        """
        is_np_input = False
        forward_inputs = OrtValueVector()
        forward_inputs.reserve(len(user_inputs))
        for tensor in user_inputs:
            if isinstance(tensor, np.ndarray):
                is_np_input = True
                forward_inputs.push_back(OrtValue.ortvalue_from_numpy(tensor)._ortvalue)
            elif isinstance(tensor, OrtValue):
                forward_inputs.push_back(tensor._ortvalue)
            else:
                raise ValueError(f"Expected input of type: numpy array or OrtValue, actual: {type(tensor)}")
        fetches = OrtValueVector()

        if self.training:
            self._model.train_step(forward_inputs, fetches)
        else:
            self._model.eval_step(forward_inputs, fetches)

        if len(fetches) == 1:
            if is_np_input:
                return fetches[0].numpy()

            return fetches[0]

        return tuple(val.numpy() for val in fetches) if is_np_input else tuple(fetches)

    def train(self, mode: bool = True) -> Module:
        """Sets the Module in training mode.

        Args:
            mode: whether to set training mode (True) or evaluation
                            mode (False). Default: True.

        Returns:
            self
        """
        self.training = mode
        return self

    def eval(self) -> Module:
        """Sets the Module in evaluation mode.

        Returns:
            self
        """
        return self.train(False)

    def lazy_reset_grad(self):
        """Lazily resets the training gradients.

        This function sets the internal state of the module such that the module gradients
        will be scheduled to be reset just before the new gradients are computed on the next invocation
        of train().
        """
        return self._model.lazy_reset_grad()

    def get_contiguous_parameters(self, trainable_only: bool = False) -> OrtValue:
        """Creates a contiguous buffer of the training session parameters

        Args:
            trainable_only: If True, only trainable parameters are considered. Otherwise, all parameters are considered.

        Returns:
            The contiguous buffer of the training session parameters.
        """
        parameters = OrtValue.ortvalue_from_shape_and_type(
            [
                self.get_parameters_size(trainable_only),
            ],
            np.float32,
            self._device_type,
            self._device.device_id(),
        )._ortvalue
        self._model.copy_parameters_to_buffer(parameters)

        return parameters

    def get_parameters_size(self, trainable_only: bool = False) -> int:
        """Returns the size of the parameters.

        Args:
            trainable_only: If True, only trainable parameters are considered. Otherwise, all parameters are considered.

        Returns:
            The number of primitive (example floating point) elements in the parameters.
        """
        return self._model.get_parameters_size(trainable_only)

    def copy_buffer_to_parameters(self, buffer: OrtValue) -> None:
        """Copies the OrtValue buffer to the training session parameters.

        Args:
            buffer: The OrtValue buffer to copy to the training session parameters.
        """
        self._model.copy_buffer_to_parameters(buffer)

    def export_model_for_inferencing(
        self, inference_model_uri: str | os.PathLike, graph_output_names: list[str]
    ) -> None:
        """Exports the model for inferencing.

        Once training is complete, this function can be used to drop the training specific nodes in the onnx model.
        In particular, this function does the following:
        - Parse over the training graph and identify nodes that generate the given output names.
        - Drop all subsequent nodes in the graph since they are not relevant to the inference graph.

        Args:
            inference_model_uri: The path to the inference model.
            graph_output_names: The list of output names that are required for inferencing.
        """
        self._model.export_model_for_inferencing(os.fspath(inference_model_uri), graph_output_names)

    def input_names(self) -> list[str]:
        """Returns the input names of the training or eval model."""
        return self._model.input_names(self.training)

    def output_names(self) -> list[str]:
        """Returns the output names of the training or eval model."""
        return self._model.output_names(self.training)
