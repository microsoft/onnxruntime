from onnxruntime.capi import _pybind_state as C


class Optimizer:
    """
    Class for running Optimize Step in Training.
    This class is a wrapper of Optimizer Class.
    """

    def __init__(self, train_optimizer_uri, model) -> None:
        """
        Initializes Optimizer with the optimizer onnx and the parameters from the model.
        """
        self._train_optimizer_uri = train_optimizer_uri
        self._model = model.get_model()
        self._optimizer = C.Optimizer(self._train_optimizer_uri, self._model)

    def step(self):
        """
        Run Optimizer Step.
        """
        self._optimizer.optimizer_step()
