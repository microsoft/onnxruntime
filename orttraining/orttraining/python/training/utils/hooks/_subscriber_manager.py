# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from collections import abc
from typing import Callable, List, Optional, Union

import torch

from onnxruntime.training.ortmodule import ORTModule

from ._subscriber_base import SubscriberBase


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


class _InspectActivation(torch.autograd.Function):
    """
    This class is used to run the subscriber's forward and backward functions.
    The function will be called by two kinds of callers:
        1. SubscriberManager calls it for each registered nn.Module.
        2. Users who want to inspect the activation tensor at any place of model definition code.
    """

    @staticmethod
    def forward(
        ctx, activation_name: str, module_idx: Optional[int], run_ctx: _RuntimeStates, input_tensor: torch.Tensor
    ):
        """
        Args:
            ctx: context object to store intermediate information.
            activation_name: the name of the activation tensor.
            module_idx:
                For call case 1 - the unique id of the module that the activation belongs to, it is detected by the
                    SubscriberManager automatically.
                For call case 2 - e.g, _InspectActivation is called by users (NOT by SubscriberManager), module_idx can
                    be None.
            run_ctx: runtime context.
                For call case 2 - need retrieve the runtime state from GlobalSubscriberManager.
            input_tensor: the activation tensor.

        Make sure there is a same number of `tensor` type inputs and outputs.
        This is enforced by ORT's PythonOp's schema check.
        """
        depth = -1
        if module_idx is not None:
            depth = run_ctx.global_states.module_index_to_depth[module_idx]

        input_tensor_copied = None
        if input_tensor is None or not isinstance(input_tensor, torch.Tensor):
            input_tensor_copied = input_tensor
        else:
            input_tensor_copied = input_tensor.detach().clone()

        ctx.current_step = run_ctx.global_states.execution_step
        ctx.name = activation_name
        ctx.id = module_idx
        ctx.depth = depth
        ctx.subscribers = run_ctx.global_states.subscribers

        # Run subscribers sequentially.
        for subscriber in run_ctx.global_states.subscribers:
            subscriber.module_post_forward(input_tensor_copied, depth, activation_name, ctx.current_step)

        return input_tensor.detach() if input_tensor is not None else None

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        val = None
        if grad_output is None or not isinstance(grad_output, torch.Tensor):
            val = grad_output
        else:
            val = grad_output.detach().clone()

        for subscriber in ctx.subscribers:
            subscriber.module_pre_backward(val, ctx.depth, ctx.name, ctx.current_step)

        return None, None, None, grad_output.detach() if grad_output is not None else None


class _IncrementStep(torch.autograd.Function):
    """
    This class is used to manage the global execution step, e.g.
    global step increment by one, once a full forward path is completed and the state clear.

    This autograd Function is registered as a post-forward hook to the root module. So once the root
    module's forward path is completed, this backward function will be called immediately, triggering
    global step increment and state clear.
    """

    @staticmethod
    def forward(ctx, run_ctx: _RuntimeStates, input_tensor: torch.Tensor):
        """
        Make sure there is a same number of `tensor` inputs and outputs.
        This is enforced by ORT's PythonOp's schema check.
        """
        ctx.current_step = run_ctx.global_states.execution_step
        ctx.run_ctx = run_ctx

        # We cannot do the step incremental here. Imagine the outside-most module has multiple outputs,
        # we need to increase the step only at the very last output handling.
        # We avoid the complexity to probe the last output handling, and instead, we assume once
        # the very first backward of the outside-most module is called, then the forward pass MUST be completed.

        # Be noted: it is not safe to register _IncrementStep only for one of the outputs of the outside-most module,
        # because we are not sure which output branch is executed earlier, for example.
        #                                   OuterMostModuleOutputs
        #                                 /                         \
        #  OuterMostModuleOutputs_0_0th_output             OuterMostModuleOutputs_0_1th_output
        #                    |                                            |
        #         PythonOp(_InspectActivation)                  PythonOp(_InspectActivation)
        #                    |                                            |
        #          PythonOp(_IncrementStep)                           graph output
        #                    |
        #                graph output
        # The PythonOp(_InspectActivation) (who relies on global step) after 1th output is possible
        # to run before or after PythonOp(_IncrementStep), so increasing the step is not safe.

        return input_tensor.detach() if isinstance(input_tensor, torch.Tensor) else input_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # In case there are multiple backward calls for multiple outputs of the outside-most module.
        if ctx.current_step == ctx.run_ctx.global_states.execution_step:
            if ctx.current_step >= 0:
                print(f"{'='*6} Completed forward pass for STEP {ctx.current_step} {'='*6}")
            ctx.run_ctx.global_states.execution_step += 1
            ctx.run_ctx.reset_step_states()

        return None, grad_output.detach() if isinstance(grad_output, torch.Tensor) else grad_output


class SubscriberManager:
    """
    This class is used to manage all the subscribers and register the post-forward hook to the root module.
    `subscribe()` is used to register a list of subscribers.

    Currently, the hook handled here is post forward hook for nn.Module. The hook is registered for all nn.Modules
    recursively. Each hook inserts a PythonOp for every tensor output generated by the corresponding module.
    Each subscriber implementation is called in the PythonOp's forward function, and backward function.

    There is one special handling for global step increment and state clear. A post-forward hook is registered
    for the outside-most module, which is the root module. In that hook, _IncrementStep is called, which will
    increase the step by 1 once the very first time its backward is called (check _IncrementStep for details).
    """

    def __init__(self):
        self._run_ctx: _RuntimeStates = _RuntimeStates()

    def subscribe(self, module: Union[torch.nn.Module, ORTModule], subscribers: List[SubscriberBase]):
        """
        The API is called externally to register hooks that are implicitly defined by subscribers.
        Each time all global states will be cleaned up once called.
        """
        if not isinstance(module, torch.nn.Module):
            raise ValueError("module must be a torch.nn.Module instance")

        self._reset_all_states()

        if isinstance(module, ORTModule):
            module = module.module

        for subscriber in subscribers:
            if not isinstance(subscriber, SubscriberBase):
                raise ValueError("subscriber must be a SubscriberBase instance")
            self._run_ctx.global_states.subscribers.add(subscriber)

        self._initialize(module)

    def get_run_context(self) -> _RuntimeStates:
        return self._run_ctx

    def _reset_all_states(self):
        self._run_ctx = _RuntimeStates()

    def _initialize(self, module: torch.nn.Module):
        """
        Register hooks for the specified module.
        """
        if len(self._run_ctx.global_states.subscribers) == 0:
            raise RuntimeError("No subscribers are registered.")

        next_module_index = [0]
        # Register post forward hook for every module, inside the hook, we loop every tensor output of the module,
        # and wrap it with an autograd Function called _InspectActivation (which takes in a tensor and returns the same
        # tensor). In this way, we keep ORT and PyTorch run have the same boundary to check activation equality.
        self._register_hooks_recursively(module, 1, next_module_index)

        # Register post forward hook for the outside-most module, then we increase the dump step.
        # Be noted, if backward is not triggered, the global dump step remains the original number,
        # which means the subsequent run will override the previous dump files. This indeed happens to imagine ORTModule
        # firstly export graph (run the forward only), after the gradient graph is built, another forward+backward is
        # triggered, override the previous dump files.
        def _post_forward_outmost_module_hook(module, _, module_outputs):
            def _apply_to_tensors_func(_, outputs):
                return _IncrementStep.apply(self._run_ctx, outputs)

            return self._apply_function_to_tensors(module, module_outputs, _apply_to_tensors_func)

        module.register_forward_hook(_post_forward_outmost_module_hook)

    def _register_hooks_recursively(self, module: torch.nn.Module, depth: int, next_module_index: List[int]):
        """
        Called to register hooks for every `torch.nn.Module`. Due to `Module` can contain child `Module`s,
        this function is called recursively by passing in `next_module_index` - a list of int to maintain a
        global incremental unique module id.

        Args:
            module: torch.nn.Module to register hook.
            depth: the indent of the module compared with the outside-most Module.
            next_module_index: list of int, carrying a global unique module index that can be used next.
        """
        module_index = next_module_index[0]
        self._run_ctx.global_states.module_index_to_depth[module_index] = depth
        self._run_ctx.global_states.module_to_module_index[module] = module_index

        for child in module.children():
            if isinstance(child, torch.nn.Module) and child not in self._run_ctx.global_states.module_to_module_index:
                next_module_index[0] += 1
                self._register_hooks_recursively(child, depth + 1, next_module_index)

        def _post_forward_module_hook(module, _, module_outputs):
            if module in self._run_ctx.global_states.module_to_module_index and isinstance(module, torch.nn.Module):
                module_index = self._run_ctx.global_states.module_to_module_index[module]

                def _apply_to_tensors_func(index, activation_tensor):
                    name = f"{module.__class__.__name__}_{module_index}_{index}th_output"
                    if name not in self._run_ctx.execution_step_states.observed_activation_names:
                        self._run_ctx.execution_step_states.observed_activation_names[name] = True
                        return _InspectActivation.apply(name, module_index, self._run_ctx, activation_tensor)

                    return activation_tensor

                return self._apply_function_to_tensors(module, module_outputs, _apply_to_tensors_func)
            return module_outputs

        module.register_forward_hook(_post_forward_module_hook)

    def _is_builtin_type(self, obj):
        # https://stackoverflow.com/a/17795199
        return obj.__class__.__module__ in ["__builtin__", "builtins"]

    def _apply_function_to_tensors(self, module: torch.nn.Module, data, func: Callable):
        """
        Apply func to all tensors in the given object.

        Args:
            module: the module that generates the tensors.
            data: the object that contains activation tensors.
            func: the function to apply to the tensors.
        """
        tensor_output_idx: List[int] = [0]

        def _apply_to_tensors_by_flatten(
            module: torch.nn.Module,
            index_for_tensor_output: List[int],
            outputs,
            func: Callable,
        ):
            if isinstance(outputs, abc.Sequence):
                touched_outputs = []
                for output in outputs:
                    touched_output = _apply_to_tensors_by_flatten(module, index_for_tensor_output, output, func)
                    touched_outputs.append(touched_output)
                return outputs.__class__(touched_outputs)

            if isinstance(outputs, abc.Mapping):
                # apply inplace to avoid recreating dict inherited objects
                for key in outputs:
                    outputs[key] = _apply_to_tensors_by_flatten(
                        module,
                        index_for_tensor_output,
                        outputs[key],
                        func,
                    )
                return outputs

            if isinstance(outputs, torch.Tensor):
                cur_id = index_for_tensor_output[0]
                index_for_tensor_output[0] += 1
                return func(cur_id, outputs)

            if not self._is_builtin_type(outputs):
                raise RuntimeError(f"Unknown type {type(outputs)}")
            return outputs

        return _apply_to_tensors_by_flatten(module, tensor_output_idx, data, func)
