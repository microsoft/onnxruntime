# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import inspect
from contextlib import contextmanager
from typing import List, Optional, Set, Tuple, Union

import onnx
import torch

from onnxruntime.training.utils import extract_data_and_schema, unflatten_data_using_schema

from ._subscriber_base import RuntimeStates, SubscriberBase

ORT_NO_INCREASE_GLOBAL_STEP = [False]


@contextmanager
def no_increase_global_step():
    """During ONNX model export phase, forward run is triggered, but we don't want to increase the global step, then
    Then the first iteration run will still start with 0, aligned with PyTorch's first iteration run.
    """
    try:
        ORT_NO_INCREASE_GLOBAL_STEP[0] = True
        yield
    finally:
        ORT_NO_INCREASE_GLOBAL_STEP[0] = False


class _IncrementStep(torch.autograd.Function):
    """This class is used to manage the global execution step, e.g.
    global step increment by one, once a full forward path is completed and the state clear.

    This autograd Function is registered as a post-forward hook to the root module. So once the root
    module's forward path is completed, this backward function will be called immediately, triggering
    global step increment and state clear.
    """

    @staticmethod
    def forward(ctx, run_ctx: RuntimeStates, *input_tensor_list: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Make sure there is the same number of `tensor` inputs and outputs.
        This is enforced by ORT's PythonOp's schema check.
        """
        ctx.current_step = run_ctx.global_states.execution_step
        ctx.run_ctx = run_ctx

        # Uncomment the following line for debugging purposes.
        # if ctx.current_step >= 0:
        #     print(f"{'='*6} Completed forward pass for STEP {ctx.current_step} {'='*6}")

        if ORT_NO_INCREASE_GLOBAL_STEP[0] is False:
            ctx.run_ctx.global_states.execution_step += 1

        return tuple(t.detach().requires_grad_(t.requires_grad) for t in input_tensor_list)

    @staticmethod
    def backward(ctx, *grad_output: Tuple[Optional[torch.Tensor], ...]) -> Tuple[Optional[torch.Tensor], ...]:
        return (None, *tuple(g for g in grad_output))

    @staticmethod
    def infer_shape(
        node: onnx.NodeProto,
        tensor_input_shapes: List[Optional[List[Union[int, str]]]],
        tensor_input_dtypes: List[torch.onnx.TensorProtoDataType],
    ) -> Tuple[List[Optional[List[Union[int, str]]]], List[torch.onnx.TensorProtoDataType]]:
        return tensor_input_shapes, tensor_input_dtypes

    @staticmethod
    def alias_input(node_proto_str: str):
        node = onnx.NodeProto()
        node.ParseFromString(node_proto_str)
        non_tensor_fw_input_count = 1
        fw_output_count = len(node.output) - 1  # exclude the first output appended in ONNX
        fw_alias_map = [-1] * fw_output_count
        bw_alias_map = [-1] * (non_tensor_fw_input_count + len(node.input))

        for i in range(fw_output_count):
            fw_alias_map[i] = i + non_tensor_fw_input_count

        tensor_input_index = 0
        for i in range(len(bw_alias_map)):
            if i < non_tensor_fw_input_count:
                continue
            bw_alias_map[i] = tensor_input_index
            tensor_input_index += 1
        return fw_alias_map, bw_alias_map


class SubscriberManager:
    """This class is used to manage all the subscribers and register subscribers' custom actions as PyTorch hooks
    to the nn.Modules.

    For the module-level/tensor_level custom actions defined by subscribers, they are registered as corresponding
    PyTorch hooks in the sequence of the subscribers' registration order.

    There is one special handling for global step increment and state clear. A post-forward hook is registered
    for the outside-most module, which is the root module. In that hook, _IncrementStep is called, which will
    increase the step by 1 once the post forward hook is called if running without no_increase_global_step().
    `no_increase_global_step` is used to skip the step increment during ONNX model export.
    """

    def __init__(self):
        self._run_ctx = RuntimeStates()
        self._subscribers: Set[SubscriberBase] = set()
        self._pre_forward_hooks = []
        self._post_forward_hooks = []

    def subscribe(self, module: torch.nn.Module, subscribers: List[SubscriberBase]):
        """
        The API is called externally to register hooks that are implicitly defined by subscribers.
        Each time all global states will be cleaned up once called.
        """
        if not isinstance(module, torch.nn.Module):
            raise ValueError("module must be a torch.nn.Module instance")

        self._reset_all_states()
        self._subscribers.clear()

        try:
            # Put the import here to avoid the module level dependency on onnxruntime.training.ortmodule
            from onnxruntime.training.ortmodule import ORTModule

            if isinstance(module, ORTModule):
                module = module.module
        except ImportError:
            pass

        for subscriber in subscribers:
            if not isinstance(subscriber, SubscriberBase):
                raise ValueError("subscriber must be a SubscriberBase instance")
            self._subscribers.add(subscriber)

        self._initialize(module)

    def get_subscriber(self, subscriber_type: type) -> SubscriberBase:
        for subscriber in self._subscribers:
            if isinstance(subscriber, subscriber_type):
                return subscriber
        raise RuntimeError(f"Subscriber {subscriber_type} is not registered.")

    def get_run_context(self) -> RuntimeStates:
        return self._run_ctx

    def _reset_all_states(self):
        self._pre_forward_hooks.clear()
        self._post_forward_hooks.clear()
        self._run_ctx = RuntimeStates()

    def _initialize(self, module: torch.nn.Module):
        """Register hooks for the specified module."""
        if len(self._subscribers) == 0:
            raise RuntimeError("No subscribers are registered.")

        def _pre_forward_outmost_module_hook(module, module_inputs):
            return _pre_forward_outmost_module_with_kwarg_hook(module, module_inputs, {})

        def _pre_forward_outmost_module_with_kwarg_hook(module, module_inputs, kwargs):
            # This check is to support the case where module is first registered in the subscriber manager,
            # then the module and hook are copied, when new module instance runs to the hook, the global states
            # are not reset, so the logic depends on the global states will fail. So in the outer-most pre-forward hook
            # we reset the global states.

            # Be noted, the first run anyway will run in PyTorch.
            if module not in self._run_ctx.global_states.module_to_module_index:
                import warnings

                warnings.warn(
                    "Initialize global states for the first time, this should only happen once for each outmost module."
                )
                self._initialize_one_time_global_states(module)

            # Call pre outmost module forward custom actions for subscribers
            for sub in self._subscribers:
                module_inputs = sub.pre_forward_outmost_module_apply(self._run_ctx, module, module_inputs, kwargs)

            return module_inputs

        # "with_kwargs" is not available for low versions of PyTorch.
        if "with_kwargs" in inspect.signature(module.register_forward_pre_hook).parameters:
            self._pre_forward_hooks.append(
                module.register_forward_pre_hook(_pre_forward_outmost_module_with_kwarg_hook, with_kwargs=True)
            )
        else:
            self._pre_forward_hooks.append(module.register_forward_pre_hook(_pre_forward_outmost_module_hook))

        next_module_index = [0]
        self._register_hooks_recursively(module, 1, next_module_index)

        # Register post forward hook for the outside-most module, then we increase the dump step.
        def _post_forward_outmost_module_hook(module, module_inputs, module_outputs):
            # Call post outmost module forward custom actions for subscribers
            for sub in self._subscribers:
                module_inputs, module_outputs = sub.post_forward_outmost_module_apply(
                    self._run_ctx, module, module_inputs, module_outputs
                )

            flatten_output_tensor_list, output_schema = extract_data_and_schema(module_outputs)
            output_tensors = _IncrementStep.apply(self._run_ctx, *flatten_output_tensor_list)
            restored_outputs = unflatten_data_using_schema(output_tensors, output_schema)

            return restored_outputs

        self._pre_forward_hooks.append(module.register_forward_hook(_post_forward_outmost_module_hook))

    def _initialize_one_time_global_states(self, module: torch.nn.Module):
        def _reset_recursively(module: torch.nn.Module, depth: int, next_module_index: List[int]):
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
            module.id = module_index  # STAGE3WARN#1: needed by DeepSpeed
            self._run_ctx.global_states.module_index_to_depth[module_index] = depth
            self._run_ctx.global_states.module_to_module_index[module] = module_index

            for child in module.children():
                if (
                    isinstance(child, torch.nn.Module)
                    and child not in self._run_ctx.global_states.module_to_module_index
                ):
                    next_module_index[0] += 1
                    _reset_recursively(child, depth + 1, next_module_index)

        next_module_index = [0]
        _reset_recursively(module, 1, next_module_index)

    def _register_hooks_recursively(self, module: torch.nn.Module, depth: int, next_module_index: List[int]):
        """Register hooks for every `torch.nn.Module`. Due to `Module` can contain child `Module`s,
        this function is called recursively by passing in `next_module_index` - a list of int to maintain a
        global incremental unique module id.

        Args:
            module: torch.nn.Module to register hook.
            depth: the indent of the module compared with the outside-most Module.
            next_module_index: list of int, carrying a global unique module index that can be used next.
        """
        module_index = next_module_index[0]
        module.id = module_index  # STAGE3WARN#2: needed by DeepSpeed
        self._run_ctx.global_states.module_index_to_depth[module_index] = depth
        self._run_ctx.global_states.module_to_module_index[module] = module_index

        for child in module.children():
            if isinstance(child, torch.nn.Module) and child not in self._run_ctx.global_states.module_to_module_index:
                next_module_index[0] += 1
                self._register_hooks_recursively(child, depth + 1, next_module_index)

        def _pre_forward_module_with_kwargs_hook(module, module_inputs, kwargs):
            # Module level hook
            for sub in self._subscribers:
                module_inputs, kwargs = sub.pre_forward_module_apply(self._run_ctx, module, module_inputs, kwargs)

            if len(self._subscribers) == 0:
                return module_inputs, kwargs

            # If there is no tensor level post forward func override, we can skip the following tensor level hook.
            if all(
                [
                    sub.__class__.pre_forward_tensor_apply_impl == SubscriberBase.pre_forward_tensor_apply_impl
                    for sub in self._subscribers
                ]
            ):
                return module_inputs, kwargs

            # Tensor level hook
            flatten_positional_input_tensor_list, input_schema = extract_data_and_schema(module_inputs)
            flatten_keyword_input_tensor_list, keyword_input_schema = extract_data_and_schema(kwargs)

            for sub in self._subscribers:
                tensor_list = []
                for tensor_index, tensor in enumerate(flatten_positional_input_tensor_list):
                    tensor_list.append(sub.pre_forward_tensor_apply(self._run_ctx, module, tensor_index, tensor))
                flatten_positional_input_tensor_list = tensor_list

                tensor_list = []
                for tensor_index, tensor in enumerate(flatten_keyword_input_tensor_list):
                    tensor_list.append(sub.pre_forward_tensor_apply(self._run_ctx, module, tensor_index, tensor))
                flatten_keyword_input_tensor_list = tensor_list

            module_inputs = unflatten_data_using_schema(flatten_positional_input_tensor_list, input_schema)
            kwargs = unflatten_data_using_schema(flatten_keyword_input_tensor_list, keyword_input_schema)

            return module_inputs, kwargs

        def _pre_forward_module_hook(module, module_inputs):
            return _pre_forward_module_with_kwargs_hook(module, module_inputs, {})

        def _post_forward_module_hook(module, module_inputs, module_outputs):
            # Module level hook
            for sub in self._subscribers:
                _, module_outputs = sub.post_forward_module_apply(self._run_ctx, module, module_inputs, module_outputs)

            if len(self._subscribers) == 0:
                return module_outputs

            # If there is no tensor level post forward func override, we can skip the following tensor level hook.
            if all(
                [
                    sub.__class__.post_forward_tensor_apply_impl == SubscriberBase.post_forward_tensor_apply_impl
                    for sub in self._subscribers
                ]
            ):
                return module_outputs

            # Tensor level hook
            flatten_output_tensor_list, output_schema = extract_data_and_schema(module_outputs)
            for sub in self._subscribers:
                tensor_list = []
                for tensor_index, tensor in enumerate(flatten_output_tensor_list):
                    tensor_list.append(sub.post_forward_tensor_apply(self._run_ctx, module, tensor_index, tensor))
                flatten_output_tensor_list = tensor_list

            return unflatten_data_using_schema(flatten_output_tensor_list, output_schema)

        # "with_kwargs" is not available for low versions of PyTorch.
        if "with_kwargs" in inspect.signature(module.register_forward_pre_hook).parameters:
            self._pre_forward_hooks.append(
                module.register_forward_pre_hook(_pre_forward_module_with_kwargs_hook, with_kwargs=True)
            )
        else:
            self._pre_forward_hooks.append(module.register_forward_pre_hook(_pre_forward_module_hook))
        self._post_forward_hooks.append(module.register_forward_hook(_post_forward_module_hook))
