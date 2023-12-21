# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import sys
from typing import Optional, Tuple

import torch

from onnxruntime.training.utils import ORTModelInputOutputType


class RuntimeStates:
    """
    A data struct holding states for runtime context.
    > Global states that are one-time collected during model hook registration. A global execution step is
      also initialized to reflect how many steps have been executed, it will get updated after each step
      completes its forward path.
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
        self.global_states = RuntimeStates._GlobalStates()


class SubscriberBase:
    """
    Base class for all module hook subscribers.

    A module hook subscriber is a class that allow define custom actions to be executed during the nn.Module's hooks.
    Two types of APIs can be used to define custom actions:
    1. Module level interfaces:
        pre_forward_module_apply - called inside the nn.Module's pre-forward hook.
        post_forward_module_apply - called inside the nn.Module's post-forward hook.
        post_forward_outmost_module_apply - called inside the nn.Module's post-forward hook, but only for the outmost module.
    2. Tensor level interfaces:
        pre_forward_tensor_apply - called inside the nn.Module's pre-forward hook, for each input tensor.
        post_forward_tensor_apply - called inside the nn.Module's post-forward hook, for each output tensor.

    For ORT runs, tensor's flows are important, that's the reason we have tensor input as function input,
    and tensor output as function output for all the APIs.
    With this, the overall flow can be traced as a data flow graph (DAG).
    """

    def __init__(self, start_step: Optional[int], end_step: Optional[int]):
        """
        Steps in [start_step, end_step) will run the subscriber's actions, and other steps will skip.
        If start_step is None, 0 is given; if end_step is None, sys.maxsize is given.
        """
        self._start_step: int = start_step if start_step is not None else 0
        self._end_step: int = end_step if end_step is not None else sys.maxsize

    def pre_forward_module_apply(
        self,
        run_ctx: RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        kwargs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        """This function is called inside the nn.Module's pre-forward hook.

        Args:
            run_ctx (RuntimeStates): The runtime states of SubscriberManager.
            module (torch.nn.Module): The module that is being executed.
            args (ORTModelInputOutputType): The positional arguments that are passed to the module's pre-forward hook.
            kwargs (ORTModelInputOutputType): The keyword arguments that are passed to the module's pre-forward hook.

        Returns:
            Tuple[ORTModelInputOutputType, ORTModelInputOutputType]: Updated args and kwargs.

        """
        if self._need_skip_step(run_ctx.global_states.execution_step):
            return args, kwargs

        updated_args, updated_kwargs = self.pre_forward_module_apply_impl(run_ctx, module, args, kwargs)
        return updated_args, updated_kwargs

    def pre_forward_module_apply_impl(
        self,
        run_ctx: RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        kwargs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        return args, kwargs

    def pre_forward_tensor_apply(
        self, run_ctx: RuntimeStates, module: torch.nn.Module, tensor_index: int, tensor: torch.Tensor
    ) -> torch.Tensor:
        """This function is called inside the nn.Module's pre-forward hook.

        Args:
            run_ctx (RuntimeStates): The runtime states of SubscriberManager.
            module (torch.nn.Module): The module that is being executed.
            tensor_index (int): The index of the tensor in the input tensor list.
            tensor (torch.Tensor): The tensor is one of module's forward inputs.
        """
        if self._need_skip_step(run_ctx.global_states.execution_step):
            return tensor

        return self.pre_forward_tensor_apply_impl(run_ctx, module, tensor_index, tensor)

    def pre_forward_tensor_apply_impl(
        self, run_ctx: RuntimeStates, module: torch.nn.Module, tensor_index: int, tensor: torch.Tensor
    ) -> torch.Tensor:
        return tensor

    def post_forward_module_apply(
        self,
        run_ctx: RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        outputs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        """This function is called inside the nn.Module's post-forward hook.

        Args:
            run_ctx (RuntimeStates): The runtime states of SubscriberManager.
            module (torch.nn.Module): The module that is being executed.
            args (ORTModelInputOutputType): The inputs arguments that are passed to the module's post-forward
                hook as input.
            outputs (ORTModelInputOutputType): The outputs arguments that are passed to the module's post-forward
                hook as input.

        Returns:
            Tuple[ORTModelInputOutputType, ORTModelInputOutputType]: Updated inputs and outputs.
        """
        if self._need_skip_step(run_ctx.global_states.execution_step):
            return args, outputs

        return self.post_forward_module_apply_impl(run_ctx, module, args, outputs)

    def post_forward_module_apply_impl(
        self,
        run_ctx: RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        outputs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        return args, outputs

    def post_forward_tensor_apply(
        self, run_ctx: RuntimeStates, module: torch.nn.Module, tensor_index: int, tensor: torch.Tensor
    ) -> torch.Tensor:
        """This function is called inside the nn.Module's post-forward hook.

        Args:
            run_ctx (RuntimeStates): The runtime states of SubscriberManager.
            module (torch.nn.Module): The module that is being executed.
            tensor_index (int): The index of the tensor in the output tensor list.
            tensor (torch.Tensor): The tensor is one of module's forward outputs.

        Returns:
            torch.Tensor: Updated tensor.
        """
        if self._need_skip_step(run_ctx.global_states.execution_step):
            return tensor

        return self.post_forward_tensor_apply_impl(run_ctx, module, tensor_index, tensor)

    def post_forward_tensor_apply_impl(
        self, run_ctx: RuntimeStates, module: torch.nn.Module, tensor_index: int, tensor: torch.Tensor
    ) -> torch.Tensor:
        return tensor

    def pre_forward_outmost_module_apply(
        self,
        run_ctx: RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        kwargs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        """This function is called inside the nn.Module's pre-forward hook.

        Args:
            run_ctx (RuntimeStates): The runtime states of SubscriberManager.
            module (torch.nn.Module): The module that is being executed.
            args (ORTModelInputOutputType): The positional arguments that are passed to the module's pre-forward hook.
            kwargs (ORTModelInputOutputType): The keyword arguments that are passed to the module's pre-forward hook.

        Returns:
            Tuple[ORTModelInputOutputType, ORTModelInputOutputType]: Updated args and kwargs.

        """
        if self._need_skip_step(run_ctx.global_states.execution_step):
            return args, kwargs

        updated_args, updated_kwargs = self.pre_forward_outmost_module_apply_impl(run_ctx, module, args, kwargs)
        return updated_args, updated_kwargs

    def pre_forward_outmost_module_apply_impl(
        self,
        run_ctx: RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        kwargs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        return args, kwargs

    def post_forward_outmost_module_apply(
        self,
        run_ctx: RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        outputs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        """This function is called inside the outmost nn.Module's post-forward hook.

        Args:
            run_ctx (RuntimeStates): The runtime states of SubscriberManager.
            module (torch.nn.Module): The module that is being executed.
            args (ORTModelInputOutputType): The inputs arguments that are passed to the module's post-forward
                hook as input.
            outputs (ORTModelInputOutputType): The outputs arguments that are passed to the module's post-forward
                hook as input.

        Returns:
            Tuple[ORTModelInputOutputType, ORTModelInputOutputType]: Updated inputs and outputs.
        """
        if self._need_skip_step(run_ctx.global_states.execution_step):
            return args, outputs

        return self.post_forward_outmost_module_apply_impl(run_ctx, module, args, outputs)

    def post_forward_outmost_module_apply_impl(
        self,
        run_ctx: RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        outputs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        return args, outputs

    def _need_skip_step(self, current_step: int) -> bool:
        return current_step < self._start_step or current_step >= self._end_step
