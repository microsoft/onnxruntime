from onnxruntime.capi import _pybind_state as C


class TrainingOptimizer:
    """
    Class for running Optimize Step in Training.
    This class is a wrapper of Optimizer Class.
    """

    def __init__(self, train_optimizer_uri, model, env=None, session_options=None, providers=None, **kwargs) -> None:
        """
        Initializes Optimizer with the optimizer onnx and the parameters from the model.
        """
        self._train_optimizer_uri = train_optimizer_uri
        self._model = model
        self._env = env
        self._session_options = session_options
        if providers is None:
            self._providers = C.get_available_providers()

        self._create_training_optimizer()

    def _create_training_optimizer(self):
        """
        This method is responsible for creating the optimizer and initializing the parameters.
        """
        # Create session options as a default value.
        self._session_options = None

        # Create session options as a default value.
        self._env = None

        # Optimizer is not exposed yet
        optimizer = C.Optimizer(self._train_optimizer_uri, self._model)

        self._optimizer = optimizer

    def step(self):
        """
        Run Optimizer Step.
        """
        return self._optimizer.optimizer_step()

    def save_checkpoint(self, ckpt_uri):
        """
        Saves the checkpoint.
        """
        return self._optimizer.save_checkpoint(ckpt_uri)

    def load_checkpoint(self, ckpt_uri):
        """
        Loads the checkpoint.
        """
        return self._optimizer.load_checkpoint(ckpt_uri)
