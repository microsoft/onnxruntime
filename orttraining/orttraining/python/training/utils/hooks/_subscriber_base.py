# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from __future__ import annotations

import sys
from typing import Tuple

import torch

from onnxruntime.training.utils import ORTModelInputOutputType


class _RuntimeStates:
    """
    A data struct holding states for runtime context. Tho kinds of states are included:
    > Global states that are one-time collected during model hook registration. A global execution step is
      also initialized to reflect how many steps have been executed, it will get updated after each step
      completes its forward path.
    > Intra-execution step states, initialized and cleaned up intended only for current execution step.
      Usually, it carries intermediate information during the model execution.
    """

    class _GlobalStates:
        def __init__(self):
            # Used to track current execution step, e.g. how many forward/backward path is called.
            self.execution_step = 0
            # Used to store the depth of each module, which indicate the indentation level of the module.
            self.module_index_to_depth = {}
            # Used to store the unique id of each sequential activation.
            self.module_to_module_index = {}

    def __init__(self):
        self.global_states = _RuntimeStates._GlobalStates()


class SubscriberBase:
    """
    Base class for all module hook subscribers.

    A module hook subscriber is a class that implements the
    `pre_forward_module_apply`, `pre_forward_tensor_apply`, `post_forward_module_apply` and `post_forward_tensor_apply`
    functions.
    1. The post_forward_* is called inside the nn.Module's post forward hook.
    2. The pre_backward_* is called inside the nn.Module's pre forward hook.

    """

    def __init__(self, start_step: None | int, end_step: None | int):
        """
        Steps in [start_step, end_step) will run the subscriber's actions, and other steps will skip.
        If start_step is None, 0 is given; if end_step is None, sys.maxsize is given.
        """
        self._start_step: int = start_step if start_step is not None else 0
        self._end_step: int = end_step if end_step is not None else sys.maxsize

    def pre_forward_module_apply(
        self,
        run_rtx: _RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        kwargs: ORTModelInputOutputType,
    ) -> tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        """This function is called inside the nn.Module's pre forward hook.

        Args:
            module (torch.nn.Module): The module that is being executed.
            run_rtx (_RuntimeStates): The runtime states of SubscriberManager.
            flatten_input_tensor_list (list[torch.Tensor]): The flattened input tensor list parsed in order from
                the pre_forward hook. Be noted: it may be parsed from positional arguments or keyword arguments.

        Returns:
            Tuple[list[torch.Tensor], list[torch.Tensor]]: The flattened input tensor list that will be passed to the module's forward.
                Usually just return from the `flatten_input_tensor_list` argument, unless we want to modify it.


        For ORTModule runs, tensor's flows are important, that's the reason we have tensor input as function input,
        and tensor output as function output. With this, the overall flow can be traced as a data flow graph (DAG).
        """
        if self._need_skip_step(run_rtx.global_states.execution_step):
            return args, kwargs

        outputs1, outputs2 = self.pre_forward_module_apply_impl(run_rtx, module, args, kwargs)
        return outputs1, outputs2

    def pre_forward_module_apply_impl(
        self,
        run_rtx: _RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        kwargs: ORTModelInputOutputType,
    ) -> tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        return args, kwargs

    def pre_forward_tensor_apply(
        self, run_rtx: _RuntimeStates, module: torch.nn.Module, tensor_index: int, tensor: torch.Tensor
    ) -> torch.Tensor:
        """This function is called inside the nn.Module's pre forward hook.

        Args:
            module (torch.nn.Module): The module that is being executed.
            run_rtx (_RuntimeStates): The runtime states of SubscriberManager.
            tensor_index (int): The index of the tensor in the input tensor list.
            tensor (torch.Tensor): The tensor which is one of module's forward inputs.
        """
        if self._need_skip_step(run_rtx.global_states.execution_step):
            return tensor

        return self.pre_forward_tensor_apply_impl(run_rtx, module, tensor_index, tensor)

    def pre_forward_tensor_apply_impl(
        self, run_rtx: _RuntimeStates, module: torch.nn.Module, tensor_index: int, tensor: torch.Tensor
    ) -> torch.Tensor:
        return tensor

    def post_forward_module_apply(
        self,
        run_rtx: _RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        output: ORTModelInputOutputType,
    ) -> tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        if self._need_skip_step(run_rtx.global_states.execution_step):
            return args, output

        return self.post_forward_module_apply_impl(run_rtx, module, args, output)

    def post_forward_module_apply_impl(
        self,
        run_rtx: _RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,  # ?? can be tensor?
        output: ORTModelInputOutputType,  # ?? can be tensor?
    ) -> tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        return args, output

    def post_forward_tensor_apply(
        self, run_rtx: _RuntimeStates, module: torch.nn.Module, tensor_index: int, tensor: torch.Tensor
    ) -> torch.Tensor:
        if self._need_skip_step(run_rtx.global_states.execution_step):
            return tensor

        return self.post_forward_tensor_apply_impl(run_rtx, module, tensor_index, tensor)

    def post_forward_tensor_apply_impl(
        self, run_rtx: _RuntimeStates, module: torch.nn.Module, tensor_index: int, tensor: torch.Tensor
    ) -> torch.Tensor:
        return tensor

    def _enforce_check(self, assert_condition: bool, msg: str = ""):
        if assert_condition is False:
            raise RuntimeError(msg)

    def _need_skip_step(self, current_step: int) -> bool:
        return current_step < self._start_step or current_step >= self._end_step

    def post_forward_outmost_module_apply(
        self,
        run_rtx: _RuntimeStates,
        module: torch.nn.Module,
        input_args: ORTModelInputOutputType,  # ?? can be tensor?
        outputs: ORTModelInputOutputType,  # ?? can be tensor?
    ) -> tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        if self._need_skip_step(run_rtx.global_states.execution_step):
            return input_args, outputs

        return self.post_forward_outmost_module_apply_impl(run_rtx, module, input_args, outputs)

    def post_forward_outmost_module_apply_impl(
        self,
        run_rtx: _RuntimeStates,
        module: torch.nn.Module,
        input_args: ORTModelInputOutputType,  # ?? can be tensor?
        outputs: ORTModelInputOutputType,  # ?? can be tensor?
    ) -> tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        return input_args, outputs
