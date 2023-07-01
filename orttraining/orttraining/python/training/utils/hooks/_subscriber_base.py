# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import sys
from typing import Optional, Union

import torch


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

            self.subscribers = set()

    class _ExecutionStepStates:
        def __init__(self):
            # Used to store the activation tensor names, if already handled, then skipped.
            # Need to clear after each step.
            self.observed_activation_names = {}

    def __init__(self):
        self.global_states = _RuntimeStates._GlobalStates()
        self.reset_step_states()

    def reset_step_states(self):
        self.execution_step_states = _RuntimeStates._ExecutionStepStates()


class SubscriberBase:
    """
    Base class for all module hook subscribers.
    Currently, the hook here only means post-forward hook and pre-backward hook.

    A module hook subscriber is a class that implements the `module_post_forward_impl` and `module_pre_backward_impl`
    function.
    > The post_forward hook is called after the activation is generated in the forward path.
    > The pre_backward hook is called before the activation gradient is computed.

    The post_forward path:
        Module_A generates activation tensor_a --> Post forward hook (calling subscribers' forward one by one) -->
        Module_B generates activation tensor_b --> ...

    The pre_backward path:
        Module_B backward run, tensor_b's gradient is computed as tensor_b_grad -->
        Pre-backward hook (calling subscribers' backward one by one) -->
        Module_A backward run, tensor_a's gradient is computed as tensor_a_grad

    Be noted: the "Pre"/"Post" is described from the perspective of Module_A.
    """

    def __init__(self, start_step: Union[None, int], end_step: Union[None, int]):
        """
        Steps in [start_step, end_step) will run the subscriber's actions, and other steps will skip.
        If start_step is None, 0 is given; if end_step is None, sys.maxsize is given.
        """
        self._start_step: int = start_step if start_step is not None else 0
        self._end_step: int = end_step if end_step is not None else sys.maxsize

    def post_forward_module_func(self, module: torch.nn.Module, module_inputs, module_outputs):
        raise NotImplementedError()

    def post_forward_tensor_func(
        self, activation_name: str, module_idx: Optional[int], run_ctx: _RuntimeStates, tensor: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    # def module_post_forward(self, activation: torch.Tensor, depth: int, name: str, step: int):
    #     """
    #     This function will be run after the torch Module forward is completed.

    #     Args:
    #         activation: Tensor to be inspected.
    #         depth: The indent level of the torch Module generating `activation`.
    #         name: The unique name for the `activation`.
    #         step: Current execution step.
    #     """
    #     if self._start_step <= step < self._end_step:
    #         self.module_post_forward_impl(activation, depth, name, step)

    # def module_pre_backward(self, activation: torch.Tensor, depth: int, name: str, step: int):
    #     """
    #     This function will be run before the torch Module backward run.

    #     Args:
    #         activation: Tensor to be inspected.
    #         depth: The indent level of the torch Module generating `activation`.
    #         name: The unique name for the `activation`.
    #         step: Current execution step.
    #     """
    #     if self._start_step <= step < self._end_step:
    #         self.module_pre_backward_impl(activation, depth, name, step)

    # def module_post_forward_impl(self, activation: torch.Tensor, depth: int, name: str, step: int):
    #     raise NotImplementedError()

    # def module_pre_backward_impl(self, activation: torch.Tensor, depth: int, name: str, step: int):
    #     raise NotImplementedError()
