# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# module.py

import numpy as np

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue, get_ort_device_type
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector


class Module:
    """
    Class for running Training.
    This class is a wrapper of Module Class.
    """

    training: bool

    def __init__(self, train_model_uri, state, eval_model_uri=None, device: str = "cpu") -> None:
        """
        Initializes Model for Training.
        __init__ will call an internatl function to create the model.
        """
        # TODO : Add support for bytes on train_model_uri and eval_model_uri.
        self.training = True
        options = device.split(":")
        self._device_type = options[0]
        device_id = 0 if len(options) < 2 else int(options[1])

        self._device = C.OrtDevice(
            get_ort_device_type(self._device_type, device_id),
            C.OrtDevice.default_memory(),
            device_id,
        )
        self._model = C.Module(train_model_uri, state._state, eval_model_uri, self._device)

    def __call__(self, user_inputs):
        """
        This method enables calling Module as a function to run the model.
        Args:
            user_inputs : list of numpy objects.
        Returns:
            fetches : list of numpy objects.
        """
        forward_inputs = OrtValueVector()
        forward_inputs.reserve(len(user_inputs))
        for element in user_inputs:
            forward_inputs.push_back(OrtValue.ortvalue_from_numpy(element)._ortvalue)
        fetches = OrtValueVector()

        if self.training:
            self._model.train_step(forward_inputs, fetches)
        else:
            self._model.eval_step(forward_inputs, fetches)

        return [val.numpy() for val in fetches]

    def train(self, mode: bool = True):
        """Sets the Module in training mode.

        This has any effect only on Module Class.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                            mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        return self

    def eval(self):
        """Sets the Module in evaluation mode.

        This has any effect only on Module Class.
        Returns:
            Module: self
        """
        return self.train(False)

    def reset_grad(self):
        """
        Resets the gradient of the parameters.
        """
        return self._model.reset_grad()

    def save_checkpoint(self, ckpt_uri):
        """
        Saves the checkpoint.
        """
        # TODO : move this out of Module Class.
        self._model.save_checkpoint(ckpt_uri)

    # This function will change when the parameters will be exposed.
    def get_contiguous_parameters(self, trainable_only: bool = False) -> OrtValue:
        """
        Returns contiguous parameters object.
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
        """
        Returns the size of the parameters.
        """
        return self._model.get_parameters_size(trainable_only)

    def copy_buffer_to_parameters(self, buffer) -> None:
        """
        Copies buffer to parameters.
        """
        self._model.copy_buffer_to_parameters(buffer)

    def export_model_for_inferencing(self, inference_model_uri: str, graph_output_names: list[str]) -> None:
        """
        Exports the model for inferencing.
        """
        self._model.export_model_for_inferencing(inference_model_uri, graph_output_names)
