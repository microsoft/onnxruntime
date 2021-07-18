from functools import lru_cache
from typing import Sequence, Tuple

from onnx import ModelProto
import torch

from ._execution_agent import InferenceAgent, TrainingAgent
from ._io import _combine_input_buffers_initializers, _InputInfo
from ._ort_module_function_factory import ort_module_function_factory
from ._utils import get_device_from_module, get_session_config

def _encode_param_name(name: str):
    """ Encode a parameter name because torch doesn't like '.' in parameter names """
    return name.replace(".", "%")

def _decode_param_name(name: str):
    """ Decode a parameter name """
    return name.replace("%", ".")


class OnnxGraphModule(torch.nn.Module):
    """Wraps inference / training onnx models and acts as a torch.nn.Module
    """
    def __init__(
        self,
        inference_model: ModelProto,
        training_model: ModelProto,
        user_input_names: Sequence[str],
        require_grad_names: Sequence[str],
        named_parameters: Sequence[Tuple[str, torch.Tensor]],
        initializer_names_to_train: Sequence[str],
        module_output_indices_requires_save_for_backward: Sequence[int] = (),
    ):
        super().__init__()
        self._inference_model = inference_model
        self._training_model = training_model
        self._user_input_names = user_input_names
        self._require_grad_names = require_grad_names
        for name, param in named_parameters:
            self.register_parameter(_encode_param_name(name), param)
        self._initializer_names_to_train = initializer_names_to_train
        self._module_output_indices_requires_save_for_backward = module_output_indices_requires_save_for_backward
        self._train = False
        self._input_info = _InputInfo(
            names=user_input_names,
            shape=[], # not accessed
            require_grad_names=require_grad_names,
            dynamic_axes={}, # not accessed
            schema={}, # not accessed
            num_positionals=len(user_input_names),
            num_positionals_non_none=len(user_input_names),
            keyword_names=[]
        )

    @lru_cache(maxsize=1)
    def inference_agent(self, device: torch.device):
        print("OnnxGraphModule: (re)-creating an InferenceAgent...")
        session_config = get_session_config(device)
        return InferenceAgent(self._inference_model, device, **session_config._asdict())

    @lru_cache(maxsize=1)
    def training_agent(self, device: torch.device):
        print("OnnxGraphModule: (re)-creating a TrainingAgent...")
        session_config = get_session_config(device)
        return TrainingAgent(self._training_model, device, **session_config._asdict())

    def train(self, mode: bool = True):
        self._train = mode
        return self

    def eval(self):
        self._train = False
        return self

    def forward(self, *args):
        """Performs forward computation
        """
        # Note that device is a property of a tensor not of a module.
        # to() method may have changed the device in between forward calls.
        device = get_device_from_module(self)
        # TODO: Can we allow kwargs?
        inputs = _combine_input_buffers_initializers(
            self.parameters(),
            self._user_input_names,
            self._input_info,
            [], # TODO: support buffers (named_buffers())
            args,
            kwargs={},
            device=device
        )
        if self._train:
            # Create a torch.autograd.Function
            _ORTModuleFunction = ort_module_function_factory(
                self.training_agent(device),
                self._user_input_names,
                self._require_grad_names,
                [_decode_param_name(name) for name, _ in self.named_parameters()],
                self._initializer_names_to_train,
                self._module_output_indices_requires_save_for_backward
            )
            outputs = _ORTModuleFunction.apply(*inputs)
        else:
            outputs, _ = self.inference_agent(device).forward(*inputs)

        # Outputs is a sequence of tensors. Handle the special case of
        # single tensor return
        if len(outputs) == 1:
            return outputs[0]
        return outputs
