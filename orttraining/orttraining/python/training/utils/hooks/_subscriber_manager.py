# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from collections import abc
from typing import Callable, List, Union

import warnings
import torch

from onnxruntime.training.ortmodule import ORTModule


class ModuleHookSubscriberBase:
    """
    Base class for all module hook subscribers.
    Currently the hook here only means post forward hook and pre backward hook.

    A module hook subscriber is a class that implements the forward and backward function.
    > The forward hook is called after the activation is generated in the forward path.
    > The backward hook is called before the activation gradient is computed.

    Forward path:
        Module_A generate activation tensor_a --> Post forward hook (calling subscribers's forward one by one) -->
        Module_B generate activation tensor_b --> ...

    Backward path:
        Module_B backward run, tensor_b's gradient is computed as tensor_b_grad -->
        Pre backward hook (calling subscribers's backward one by one) -->
        Module_A backward run, tensor_a's gradient is computed as tensor_a_grad

    Be noted: the "Pre"/"Post" is described from the perspective of the Module_A.
    """

    def __init__(self, start_step: Union[None, int], end_step: Union[None, int]):
        """
        Steps in [start_step, end_step) will run subscriber's actions, other steps will skip.
        """
        self.start_step = start_step
        self.end_step = end_step

    def module_post_forward(self, activation: torch.Tensor, depth: int, name: str, step: int):
        """
        This function will be run after the torch Module forward completed.
        :param activation: Tensor to be inspected.
        :param depth: The indend level of the torch Module generating `activation`.
        :param name: The unique name for the `activation`.
        :param step: Current execution step.
        """
        if self.start_step is not None and self.end_step is not None:
            if self.start_step <= step < self.end_step:
                self.module_post_forward_impl(activation, depth, name, step)

    def module_pre_backward(self, activation: torch.Tensor, depth: int, name: str, step: int):
        """
        This function will be run before the torch Module backward run.
        :param activation: Tensor to be inspected.
        :param depth: The indent level of the torch Module generating `activation`.
        :param name: The unique name for the `activation`.
        :param step: Current execution step.
        """
        if self.start_step is not None and self.end_step is not None:
            if self.start_step <= step < self.end_step:
                self.module_pre_backward_impl(activation, depth, name, step)

    def module_post_forward_impl(self, activation: torch.Tensor, depth: int, name: str, step: int):
        raise NotImplementedError()

    def module_pre_backward_impl(self, activation: torch.Tensor, depth: int, name: str, step: int):
        raise NotImplementedError()


class _RuntimeStates:
    """
    A data struct holding states for runtime context. Tho kinds of states are included:
    > Global states that is one-time collected during model hook registration. A global execution step is
      also initialized to reflect how many steps have been executed.
    > Intra execution step states, initialized and cleaned up intended only for current execution step.
      Usually it carriers intermediate informations during the model execution.
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
    This class is used to run the subscriber's forward and backward function.
    It also manages step related checks, including run subscribers only for between start_step and end_step.
    """

    @staticmethod
    def forward(ctx, activation_name: str, module_idx: int, run_ctx: _RuntimeStates, input_tensor):
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

        # Run subscribers squentially.
        for subscriber in run_ctx.global_states.subscribers:
            subscriber.module_post_forward(input_tensor_copied, depth, activation_name, ctx.current_step)

        return input_tensor.detach() if input_tensor is not None else None

    @staticmethod
    def backward(ctx, grad_output):
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
    global step increment by one, once a full forward path completed and state clear.

    This autograd Function is registered as a post forward hook to the root module. So once the root
    module's forward path is completed, this backward function will be called immediately, triggering
    global step increment and state clear.
    """

    @staticmethod
    def forward(ctx, run_ctx, input_tensor):
        ctx.current_step = run_ctx.global_states.execution_step
        ctx.run_ctx = run_ctx

        # We cannot do the step incremental here. Imagine the outside-most module has multiple outputs,
        # we need increase the step only at the very last output handling.
        # We avoid the complexity to probe the last output handling, and instead, we assume once
        # the very first backward of outside-most module is called, then the forward pass MUST be completed.

        # Be noted: it is not safe to register _IncrementStep only for any of the outpus of outer-most mododule,
        # because we are not sure which output branch is executed, for example.
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
    def backward(ctx, grad_output):
        if ctx.current_step >= 0:
            print(f"{'='*6} Completed forward pass for STEP {ctx.current_step} {'='*6}")
        ctx.run_ctx.global_states.execution_step += 1
        ctx.run_ctx.reset_step_states()

        return None, grad_output.detach() if isinstance(grad_output, torch.Tensor) else grad_output


class SubscriberManager:
    """
    This class is used to manage all the subscribers, and register post forward hook to the root module.
    _register_subscriber() is used to register a subscriber, and can be chained with other register calls.

    Currently, the hook handled here is post forward hook for nn.Module. Hook is registered for all nn.Modules
    recursively. Each hook insert a PythonOp for every tensor output generated by the corresponding module.
    Each subscriber implementation is called in the PythonOp's forward function, and backward function.

    There is one special handling for global step increment and state clear. A post forward hook is registered
    for the outside-most module, which is the root module. In that hook, _IncrementStep is called, which will
    increase the step by 1 once the very first time its backward is called (check _IncrementStep for details).
    """

    def __init__(self):
        self._run_ctx: _RuntimeStates = _RuntimeStates()

    def subscribe(self, module: Union[torch.nn.Module, ORTModule], subscribers: List[ModuleHookSubscriberBase]):
        """
        The API called externally to register hooks that is implicitly defined by subscribers.
        Each time all global states will be cleaned up once called.
        """
        if not isinstance(module, torch.nn.Module):
            raise ValueError("module must be a torch.nn.Module instance")

        self._reset_all_states()

        if isinstance(module, ORTModule):
            module = module.module

        for subscriber in subscribers:
            if not isinstance(subscriber, ModuleHookSubscriberBase):
                raise ValueError("subscriber must be a ModuleHookSubscriberBase instance")
            self._run_ctx.global_states.subscribers.add(subscriber)

        self._initialize(module)

    def _reset_all_states(self):
        self._run_ctx = _RuntimeStates()

    def _initialize(self, module: torch.nn.Module):
        """
        Register hools for specified module.
        """
        if len(self._run_ctx.global_states.subscribers) == 0:
            warnings.warn("No subscribers are registered, skip insert hooks.")
            return

        next_module_index = [0]
        # Register post forward hook for every module, inside the hook, we loop every tensor output of the module,
        # and wrap it with a autograd Function called _InspectActivation (which take in a tensor and return the same
        # tensor). In this way, we keep ORT and PyTorch run have the same boundary to check activation equality.
        self._register_hooks_recursively(module, 1, next_module_index)

        # Register post forward hook for outmost module, then we increase the dump step.
        # Be noted, if backward is not triggered, the global dump step remain the original number,
        # which means the subsequent run will override the previous dump files. This indeed happens imagine ORTModule
        # firstly export graph (run the forward only), after gradient graph is built, another forward+backward is
        # triggered, override the previous dump files.
        def _post_forward_outmost_module_hook(module, _, module_outputs):
            def _apply_to_tensors_func(_, outputs):
                return _IncrementStep.apply(self._run_ctx, outputs)

            return self._apply_function_to_tensors(
                module, module_outputs, _apply_to_tensors_func, first_differentiable_tensor_only=True
            )

        module.register_forward_hook(_post_forward_outmost_module_hook)

    def _register_hooks_recursively(self, module: torch.nn.Module, depth: int, next_module_index: List[int]):
        """
        Called to register hooks for every `torch.nn.Module`. Due to `Module` can contain child `Module`s,
        this function is called recursively by passing in `next_module_index` - a list of int to maintain a
        global incremental unique module id.
        :param module: torch.nn.Module to register hook.
        :param depth: the indent of module compared with the outer-most Module.
        :param next_module_index: list of int, carrying a global unique module index that can be used next.
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

    def _apply_function_to_tensors(
        self, module: torch.nn.Module, data, func: Callable, first_differentiable_tensor_only: bool = False
    ):
        """
        Apply func to all tensors in the given object.
        :param module: the module that generates the tensors.
        :param data: the object that contains activation tensors.
        :param func: the function to apply to the tensors.
        :param first_differentiable_tensor_only: only apply the `func` for the first differentiable tensor only.
        """
        tensor_output_idx: List[int] = [0]
        first_differentiable_tensor_output: List[int] = [-1]

        def _apply_to_tensors_by_flatten(
            module: torch.nn.Module,
            index_for_tensor_output: List[int],
            outputs,
            func: Callable,
            first_differentiable_tensor_output: List[int],
            first_differentiable_tensor_only: bool = False,
        ):
            if isinstance(outputs, abc.Sequence):
                touched_outputs = []
                for output in outputs:
                    touched_output = _apply_to_tensors_by_flatten(
                        module,
                        index_for_tensor_output,
                        output,
                        func,
                        first_differentiable_tensor_output,
                        first_differentiable_tensor_only,
                    )
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
                        first_differentiable_tensor_output,
                        first_differentiable_tensor_only,
                    )
                return outputs

            if isinstance(outputs, torch.Tensor):
                cur_id = index_for_tensor_output[0]
                index_for_tensor_output[0] += 1
                if first_differentiable_tensor_only is True and first_differentiable_tensor_output[0] >= 0:
                    return outputs
                else:
                    first_differentiable_tensor_output[0] = cur_id
                    return func(cur_id, outputs)

            if not self._is_builtin_type(outputs):
                raise RuntimeError(f"Unknown type {type(outputs)}")
            return outputs

        return _apply_to_tensors_by_flatten(
            module, tensor_output_idx, data, func, first_differentiable_tensor_output, first_differentiable_tensor_only
        )
