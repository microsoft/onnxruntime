from onnxruntime.capi import _pybind_state as C


class TrainingOptimizer:

    def __init__(self, train_optimizer_uri , model , env=None, session_options=None, providers=None, **kwargs) -> None:
        """
        Initializes Model for Training.
        """
        self._train_optimizer_uri = train_optimizer_uri
        self._model = model
        self._env = env
        self._session_options = session_options
        if providers is None:
            self._providers = C.get_available_providers()


        self._create_training_optimizer()



    def _create_training_optimizer(self):
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
        return self._optimizer.step()
