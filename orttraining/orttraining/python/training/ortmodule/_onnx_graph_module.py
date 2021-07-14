from typing import Dict, Iterator, Optional, Sequence, Tuple, Union

from onnx import ModelProto
import torch

from onnxruntime import SessionOptions
from ._execution_agent import InferenceAgent, TrainingAgent
from ._io import _combine_input_buffers_initializers, _InputInfo
from ._ort_module_function_factory import ort_module_function_factory


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
        session_options: Optional[SessionOptions] = None,
        providers: Optional[Sequence[Union[str, Tuple[str, Dict]]]] = None,
        provider_options: Optional[Sequence[Dict]] = None
    ):
        super().__init__()
        self._device = torch.device("cpu")
        self._inference_agent = InferenceAgent(inference_model, self._device, session_options, providers, provider_options)
        self._training_agent = TrainingAgent(training_model, self._device, session_options, providers, provider_options)
        self._user_input_names = user_input_names
        self._require_grad_names = require_grad_names
        self._named_parameters = named_parameters
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

    def train(self, mode: bool = True):
        self._train = mode
        return self

    def eval(self):
        self._train = False
        return self

    def to(self, device: torch.device): # pylint: disable=arguments-differ
        assert isinstance(device, torch.device), "Currently the only operation supported is device movement."
        super().to(device)
        self._inference_agent.device = device
        self._training_agent.device = device
        return self

    def parameters(self, _recurse: bool = True) -> Iterator[torch.Tensor]:
        return (param for _, param in self._named_parameters)

    def named_parameters(self, prefix: str = "", _recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        return ((f"{prefix}{name}", param) for name, param in self._named_parameters)

    def forward(self, *args):
        """Performs forward computation
        """
        # TODO: Can we allow kwargs?
        inputs = _combine_input_buffers_initializers(
            self.parameters(),
            self._user_input_names,
            self._input_info,
            [], # TODO: support buffers (named_buffers())
            args,
            kwargs={},
            device=self._device
        )
        if self._train:        
            # Create a torch.autograd.Function
            _ORTModuleFunction = ort_module_function_factory(
                self._training_agent,
                self._user_input_names,
                self._require_grad_names,
                [name for name, _ in self.named_parameters()],
                self._initializer_names_to_train,
                self._module_output_indices_requires_save_for_backward
            )
            outputs = _ORTModuleFunction.apply(*inputs)
        else:
            outputs, _ = self._inference_agent.forward(*inputs)

        # Outputs is a sequence of tensors. Handle the special case of
        # single tensor return
        if len(outputs) == 1:
            return outputs[0]
        return outputs
