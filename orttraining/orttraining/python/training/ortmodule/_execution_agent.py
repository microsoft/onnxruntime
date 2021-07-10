# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from onnx import ModelProto
from onnx.helper import get_attribute_value
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import IOBinding, get_ort_device_type
from onnxruntime.capi._pybind_state import TrainingAgent as C_TrainingAgent

from . import _utils
from ._graph_execution_manager import RunStateInfo

import torch
from torch.utils.dlpack import to_dlpack

class ExecutionAgentOutput(object):
    def __init__(self, ortvalues, run_id=None):
        self.ortvalues = ortvalues
        self.run_id = run_id

class ExecutionAgent(object):
    """ExecutionAgent wraps ORT inference session and provides forward / backward methods
    for ORTModule and users who would like to run an ONNX graph with torch inputs
    """

    def __init__(self, onnx_model: ModelProto, device: torch.device):
        """Initializes ExecutionAgent

        Args:
        :param onnx_model: ONNX ModelProto
        :param device: torch device
        """
        self._onnx_model = onnx_model
        self._device = device

    @abstractmethod
    def forward(self, *inputs: Sequence[torch.Tensor]) -> Tuple[Sequence[torch.Tensor], RunStateInfo]:
        """Runs forward computation
        """

class InferenceAgent(ExecutionAgent):
    """
    This is the main class used to run an ORTModule model inferencing.
    """

    def __init__(self, onnx_model, device, session_options=None, providers=None, provider_options=None):
        """
        :param onnx_model: ONNX ModelProto
        :param device: torch device
        :param sess_options: session options
        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.

        The model type will be inferred unless explicitly set in the SessionOptions.
        To explicitly set:
          so = onnxruntime.SessionOptions()
          so.add_session_config_entry('session.load_model_format', 'ONNX') or
          so.add_session_config_entry('session.load_model_format', 'ORT') or

        A file extension of '.ort' will be inferred as an ORT format model.
        All other filenames are assumed to be ONNX format models.

        'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

        The list of providers is ordered by precedence. For example ['CUDAExecutionProvider', 'CPUExecutionProvider']
        means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.
        """
        super().__init__(onnx_model, device)
        self._inference_session = None

        self.create_inference_agent(onnx_model.SerializeToString(), session_options, providers, provider_options)

    def create_inference_agent(self, bytes_, session_options, providers, provider_options):
        self._inference_session = onnxruntime.InferenceSession(bytes_, session_options,
                                                               providers, provider_options)

    def io_binding(self):
        """Return an onnxruntime.IOBinding object`."""

        return IOBinding(self._inference_session)

    def _process_inputs(self, inputs):
        """Runs the forward graph on execution_session with given model inputs and device"""

        # Assert that the input and model device match
        _utils._check_same_device(self._device, "Input argument to forward", *inputs)

        # TODO: Try to reuse the output buffers as some of the output tensors are same sizes,
        #   especially the backward graph outputs.
        # REVIEW(codemzs): Consolidate Training Agent with InferenceAgent on C++ side to not
        # have the need for passing IOBinding.
        io_binding = IOBinding(self._inference_session)


        # Use IO binding
        _utils._create_iobinding(io_binding, inputs, self._onnx_model, self._device)

        return io_binding

    def _process_outputs(self, ort_output):
        forward_outputs = ort_output.ortvalues
        user_outputs = tuple(_utils._ortvalue_to_torch_tensor(forward_output._ortvalue) for forward_output in forward_outputs)
        state = None

        # Assert that the outputs and model device match
        _utils._check_same_device(self._device, "Output argument from forward", *user_outputs)

        output_info = [(output.shape, output.device, output.dtype) for output in user_outputs]
        run_info = RunStateInfo(state, output_info)
        # Return user outputs and forward run information
        return user_outputs, run_info

    def _run_forward(self, iobinding, run_options):
        """
         Compute the forward graph.
         :param iobinding: the iobinding object that has graph inputs/outputs bind.
         :param run_options: See :class:`onnxruntime.RunOptions`.
        """

        self._inference_session.run_with_iobinding(iobinding, run_options)
        ortvalues = iobinding.get_outputs()
        return ExecutionAgentOutput(ortvalues)

    def forward(self, *inputs: Sequence[torch.Tensor]) -> Tuple[Sequence[torch.Tensor], RunStateInfo]:
        run_options = C.RunOptions()
        io_binding = self._process_inputs(inputs)

        # Run and return module outputs.
        ort_output = self._run_forward(io_binding, run_options)

        return self._process_outputs(ort_output)


@dataclass(frozen=True)
class YieldOpInfo:
    """Minimum version of GraphInfo needed for TrainingAgent
    """
    user_output_names: List[str]
    non_differentiable_outputs: List[int]
    full_shape_outputs: List[int]

    @staticmethod
    def from_training_model(onnx_model):
        yield_op = next(op for op in onnx_model.graph.node if op.op_type == "YieldOp")
        attrs = {
            attr.name: get_attribute_value(attr)
            for attr in yield_op.attribute
        }
        return YieldOpInfo(
            [output for output in yield_op.input],
            attrs.get("non_differentiable_outputs", []),
            attrs.get("full_shape_outputs", [])
        )

class TrainingAgent(ExecutionAgent):
    """
    This is the main class used to run an ORTModule model training.
    """

    def __init__(self, onnx_model, device, session_options=None,
                 providers=None, provider_options=None):
        """
        :param onnx_model: ONNX ModelProto
        :param device: torch device
        :param sess_options: session options
        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.

        The model type will be inferred unless explicitly set in the SessionOptions.
        To explicitly set:
          so = onnxruntime.SessionOptions()
          so.add_session_config_entry('session.load_model_format', 'ONNX') or
          so.add_session_config_entry('session.load_model_format', 'ORT') or

        A file extension of '.ort' will be inferred as an ORT format model.
        All other filenames are assumed to be ONNX format models.

        'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

        The list of providers is ordered by precedence. For example ['CUDAExecutionProvider', 'CPUExecutionProvider']
        means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.
        """
        super().__init__(onnx_model, device)
        self._info = YieldOpInfo.from_training_model(onnx_model)
        self._inference_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), session_options,
                                                               providers, provider_options)

        fw_feed_names = [input.name for input in onnx_model.graph.input]
        fw_outputs_device_info = [
            C.OrtDevice(get_ort_device_type(self._device.type),
                        C.OrtDevice.default_memory(),
                        _utils.get_device_index(self._device)
            )] * len(self._info.user_output_names)

        bw_fetches_names = [output.name for output in onnx_model.graph.output]
        bw_outputs_device_info = [
            C.OrtDevice(get_ort_device_type(self._device.type),
                        C.OrtDevice.default_memory(),
                        _utils.get_device_index(self._device)
            )] * len(bw_fetches_names)
        self._training_agent = C_TrainingAgent(self._inference_session._sess, fw_feed_names, fw_outputs_device_info,
                                               bw_fetches_names, bw_outputs_device_info)



    def _process_forward_inputs(self, inputs):
        """ Prepare feeds from torch tensors """

        # Assert that the input and model device match
        _utils._check_same_device(self._device, "Input argument to forward", *inputs)
        # TODO: Try to reuse the output buffers as some of the output tensors are same sizes,
        #   especially the backward graph outputs.
        # REVIEW(codemzs): Consolidate Training Agent with InferenceAgent on C++ side to not
        # have the need for passing IOBinding.
        forward_inputs = C.OrtValueVector()
        forward_inputs.reserve(len(inputs))
        for input in inputs:
            forward_inputs.push_back(to_dlpack(input), input.dtype == torch.bool)

        return forward_inputs


    def _process_forward_outputs(self, forward_outputs, state):
        user_outputs = tuple(_utils._ortvalue_to_torch_tensor(forward_output) for forward_output in forward_outputs)
        output_info = [(output.shape, output.device, output.dtype) for output in user_outputs]
        run_info = RunStateInfo(state, output_info)
        # Return user outputs and forward run information
        return user_outputs, run_info


    def forward(self, *inputs: Sequence[torch.Tensor]) -> Tuple[Sequence[torch.Tensor], RunStateInfo]:
        feeds = self._process_forward_inputs(inputs)
        fetches = C.OrtValueVector()
        state = C.PartialGraphExecutionState()
        self._training_agent.run_forward(feeds, fetches, state)
        return self._process_forward_outputs(fetches, state)


    def _process_backward_inputs(self, run_info, grad_outputs):
        _utils._check_same_device(self._device, "Input argument to backward", *grad_outputs)

        # Use IO binding
        # Push user output grads to ONNX backend.
        backward_inputs = C.OrtValueVector()
        # Preallocate length of the vector. And then delete as required towards the end.
        backward_inputs.reserve(len(grad_outputs))
        for idx, grad_output in enumerate(grad_outputs):
            if idx in self._info.non_differentiable_outputs:
                assert grad_output is None, "ORT found the {}-th module output '{}' is " \
                                            "non-differentiable according to the onnx graph. " \
                                            "However, the gradient value is still provided by " \
                                            "PyTorch's autograd engine." \
                                            .format(idx, self._info.user_output_names[idx])
                continue

            if grad_output is None:
                shape, device, dtype = run_info.output_info[idx]
                if idx in self._info.full_shape_outputs:
                    grad_output = torch.zeros(shape, device=device, dtype=dtype)
                else:
                    grad_output = torch.tensor(0., device=device, dtype=dtype)
            elif not grad_output.is_contiguous():
                grad_output = grad_output.contiguous()
            backward_inputs.push_back(to_dlpack(grad_output), grad_output.dtype == torch.bool)
        backward_inputs.shrink_to_fit()
        return backward_inputs


    @staticmethod
    def _process_backward_outputs(backward_outputs):
        return [
            _utils._torch_tensor_from_dl_pack(backward_outputs.dlpack_at(i), backward_output)
            for i, backward_output in enumerate(backward_outputs)
        ]


    def backward(self, run_info, *grad_outputs):
        backward_inputs = self._process_backward_inputs(run_info, grad_outputs)
        # Run and get results
        backward_outputs = C.OrtValueVector()
        self._training_agent.run_backward(backward_inputs, backward_outputs, run_info.state)
        return self._process_backward_outputs(backward_outputs)
