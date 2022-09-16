# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# module.py

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector


class Module:
    """
    Class for running Training.
    This class is a wrapper of Module Class.
    """

    training: bool

    def __init__(self, train_model_uri, state, eval_model_uri=None) -> None:
        """
        Initializes Model for Training.
        __init__ will call an internatl function to create the model.
        """
        # TODO : Add support for bytes on train_model_uri and eval_model_uri.
        self.training = True
        self._model = C.Module(train_model_uri, state._state, eval_model_uri)

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
