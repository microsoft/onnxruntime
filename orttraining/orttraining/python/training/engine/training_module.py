from onnxruntime.capi import _pybind_state as C


class TrainingModule:

    def __init__(self, train_model_uri, ckpt_uri ,eval_model_uri=None, env=None, session_options=None, providers=None, **kwargs) -> None:
        """
        Initializes Model for Training.
        """
        # TODO : Add support for bytes on train_model_uri and eval_model_uri.
        self._train_model_uri = train_model_uri
        self._ckpt_uri = ckpt_uri
        self._eval_model_uri = eval_model_uri
        self._env = env
        self._session_options = session_options
        if providers is None:
            self._providers = C.get_available_providers()


        self._create_training_module()



    def _create_training_module(self):

        # Load checkpoint to state, then Pull named parameters from module_checkpoint_state to _parameters.
        self._parameters = {}

        # Create session options as a default value.
        self._session_options = None

        # Create session options as a default value.
        self._env = None
        # Module is not exposed yet
        model = C.Module(self._train_model_uri, self._ckpt_uri)
        self._model = model


    def train(self, input, fetches):
        """
        Trains the model.
        """
        return self._model.train(input, fetches)


    def eval(self, input, fetches):
        """
        Evaluates the model.
        """
        return self._model.eval(input, fetches)

    def parameters(self):
        """
        Returns the parameters of the model.
        Typically passed to the optimizer.
        """
        return self._model.parameters

    def get_model(self):
        """
        Returns the model to be passed to initialize the Optimizer.
        """
        return self._model
