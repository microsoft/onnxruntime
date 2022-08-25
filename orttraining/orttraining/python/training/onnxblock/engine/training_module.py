import numpy as np

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector


class TrainingModule:
    """
    Class for running Training.
    This class is a wrapper of Module Class.
    """
    training: bool
    def __init__(
        self, train_model_uri, ckpt_uri, eval_model_uri=None, env=None, session_options=None, providers=None, **kwargs
    ) -> None:
        """
        Initializes Model for Training.
        __init__ will call an internatl function to create the model.
        """
        # TODO : Add support for bytes on train_model_uri and eval_model_uri.
        self.training = True
        self.fetches = OrtValueVector()
        self._train_model_uri = train_model_uri
        self._ckpt_uri = ckpt_uri
        self._eval_model_uri = eval_model_uri
        self._env = env
        self._session_options = session_options
        if providers is None:
            self._providers = C.get_available_providers()

        self._create_training_module()

    def __call__(self, *input):

        if(self.training):
            self._model.train_step(*input,self.fetches)
        elif(self.training == False):
            self._model.eval_step(*input,self.fetches)

        return self.fetches[0].numpy()

    def _create_training_module(self):
        """
        This method is responsible for creating the model and initializing the parameters.
        """
        # Create session options as a default value.
        self._session_options = None

        # Create session options as a default value.
        self._env = None
        # Module is not exposed yet
        model = C.Module(self._train_model_uri, self._ckpt_uri, self._eval_model_uri = eval_model_uri)
        self._model = model

    def train(self, mode : bool = True):
        """Sets the TrainingModule in training mode.

        This has any effect only on TrainingModule Class.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                            mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        return self

    def eval(self):
        """Sets the TrainingModule in evaluation mode.

        This has any effect only on TrainingModule Class.
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
        Returns the model to be passed to initialize the Optimizer.
        """
        return self._model

    def parameters(self):
        """
        Returns the parameters of the model.
        """
        return self._model.parameters()

    def loss(self):
        """
        Returns the loss value.
        """
        if(len(self.fetches) == 0):
            raise ValueError("You should run a train step before calling this function.")
        return self.fetches[0].numpy()
