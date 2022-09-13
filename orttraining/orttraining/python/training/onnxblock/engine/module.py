import torch

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector


class Module:
    """
    Class for running Training.
    This class is a wrapper of Module Class.
    """

    training: bool

    def __init__(self, train_model_uri, ckpt_uri, eval_model_uri=None) -> None:
        """
        Initializes Model for Training.
        __init__ will call an internatl function to create the model.
        """
        # TODO : Add support for bytes on train_model_uri and eval_model_uri.
        self.training = True
        self._train_model_uri = train_model_uri
        self._ckpt_uri = ckpt_uri
        self._eval_model_uri = eval_model_uri

        self._create_training_module()

    def __call__(self, *input):
        """
        This method enables calling Module as a function to run the model.
        Args:
            input (OrtValueVector): input vector of ortvalues.
        Returns:
            fetches : OrtValueVector that has model output.
        """
        fetches = OrtValueVector()
        if self.training:
            self._model.train_step(*input, fetches)
        else:
            self._model.eval_step(*input, fetches)

        return fetches

    def _create_training_module(self):
        """
        This method is responsible for creating the model and initializing the parameters.
        """
        # TODO : make this as a util function.
        self._model = C.Module(self._train_model_uri, self._ckpt_uri, self._eval_model_uri)

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

    def get_model(self):
        """
        Returns the module class to be passed to initialize the Optimizer.
        """
        return self._model

    def get_contagious_parameters(self):
        """
        Returns an ORTvalue that contains the buffer output of the Module's parameters.
        """

        arr = torch.zeros(self._model.get_parameters_size(False)).numpy()
        output = C.OrtValue.ortvalue_from_numpy(
            arr,
            C.OrtDevice(
                C.OrtDevice.cpu(),
                C.OrtDevice.default_memory(),
                0,
            ),
        )

        self._model.copy_parameters_to_buffer(output)

        return output

    def save_checkpoint(self, ckpt_uri):
        """
        Saves the checkpoint.
        """
        # TODO : move this out of Module Class.
        self._model.save_checkpoint(ckpt_uri)
