# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import inspect
from types import FunctionType
from typing import Tuple

import torch

# The hooks reference functions or classes in that file.
from deepspeed.runtime.zero.parameter_offload import *
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.utils import instrument_w_nvtx

from onnxruntime.training.utils import ORTModelInputOutputType, extract_data_and_schema, unflatten_data_using_schema

from ._subscriber_base import SubscriberBase, _RuntimeStates

ds_func_store = {}


def collect_hook_functions(offloader: DeepSpeedZeRoOffload):
    code_obj = DeepSpeedZeRoOffload._register_hooks_recursively.__code__
    for c in code_obj.co_consts:
        if inspect.iscode(c):
            ds_func_store[c.co_name] = [offloader, c, False]

    code_obj = DeepSpeedZeRoOffload.setup_zero_stage3_hooks.__code__
    for c in code_obj.co_consts:
        if inspect.iscode(c):
            ds_func_store[c.co_name] = [offloader, c, False]


# Used to monkey patch the original function
# Adapted from https://github.com/microsoft/DeepSpeed/blob/e8318634b4313eaad89842cf4322e1762d34ced3/deepspeed/runtime/zero/parameter_offload.py#L333
def setup_zero_stage3_ort_compatible_hooks(self):
    self.hierarchy = 0

    from onnxruntime.training.utils.hooks import GlobalSubscriberManager, ZeROOffloadSubscriber
    from onnxruntime.training.utils.hooks._zero_offload_subscriber import collect_hook_functions

    collect_hook_functions(self)
    GlobalSubscriberManager.subscribe(self.module, [ZeROOffloadSubscriber(self)])
    self.forward_hooks.extend(GlobalSubscriberManager._pre_forward_hooks)
    self.forward_hooks.extend(GlobalSubscriberManager._post_forward_hooks)

    # Add top module to stack trace
    global FWD_MODULE_STACK
    FWD_MODULE_STACK.append(self.module)


DeepSpeedZeRoOffload.setup_zero_stage3_hooks = setup_zero_stage3_ort_compatible_hooks


def _create_closure_for_ds_hook_function(offloader):
    # https://stackoverflow.com/questions/17395338/how-to-access-a-function-inside-a-function
    def make_closure_cell(_self):
        def nested():
            return _self

        return nested.__closure__[0]

    cell = make_closure_cell(offloader)
    return cell


class ORTZeROOffloadPreForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module,
        pre_forward_function,
        post_backward_function,
        args_schema,
        kwargs_schema,
        args_tensors,
        kwargs_tensors,
        partitioned_params,
    ):
        args = unflatten_data_using_schema(args_tensors, args_schema)
        kwargs = unflatten_data_using_schema(kwargs_tensors, kwargs_schema)
        f_ret = pre_forward_function(module, args, kwargs)

        if f_ret is None:
            updated_args, updated_kwargs = args, kwargs
        else:
            assert isinstance(f_ret, tuple)
            updated_args, updated_kwargs = f_ret

        # Moved from _post_backward_module_hook to make sure ORT run will trigger every iteration.
        module.ds_grads_remaining = 0

        ctx.post_backward_function = post_backward_function
        ctx.args_count = len(args_tensors)
        ctx.kwargs_count = len(kwargs_tensors)
        ctx.partitioned_params_count = len(partitioned_params)

        updated_args_tensors, _ = extract_data_and_schema(updated_args)
        updated_kwargs_tensors, _ = extract_data_and_schema(updated_kwargs)

        rets = updated_args_tensors + updated_kwargs_tensors
        rets += tuple([p.detach().requires_grad_(p.requires_grad) for p in partitioned_params])

        # PyTorch exporter does not support an empty list of tensors, so we have this check.
        assert len(rets) != 0
        return rets

    @staticmethod
    def backward(ctx, *args):
        return (None, None, None, None, None, *args)


class ORTZeROOffloadPostForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module,
        post_forward_function,
        pre_backward_function,
        output_schema,
        *output_tensors,
    ):
        outputs = unflatten_data_using_schema(output_tensors, output_schema)

        # WARN: _post_forward_module_hook's second argument `input is not used, so we just pass a None here.
        updated_outputs = post_forward_function(module, None, outputs)

        if updated_outputs is None:
            updated_outputs = outputs

        updated_output_tensors, _ = extract_data_and_schema(updated_outputs)

        ctx.module = module
        ctx.pre_backward_function = pre_backward_function

        return tuple([o.detach().requires_grad_(o.requires_grad) for o in updated_output_tensors])

    @staticmethod
    def backward(ctx, *args):
        return (None, None, None, None, *args)


class ZeROOffloadSubscriber(SubscriberBase):
    """This subscriber is used to enable ZeRO Offload feature in a way compatible with ORTModule."""

    def __init__(self, offloader: DeepSpeedZeRoOffload):
        super().__init__(None, None)
        self._offloader = offloader

    def pre_forward_module_apply_impl(
        self,
        run_rtx: _RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        kwargs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        ds_func_config = ds_func_store["_pre_forward_module_hook"]
        if ds_func_config[2] is False:
            # initialize the hook func.
            c = ds_func_config[1]
            cell = _create_closure_for_ds_hook_function(ds_func_config[0])
            ds_func_config[1] = FunctionType(c, globals(), c.co_name, None, (cell,))
            ds_func_config[2] = True

        args_tensors, args_schema = extract_data_and_schema(args)
        kwargs_tensors, kwargs_schema = extract_data_and_schema(kwargs)

        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        from deepspeed.runtime.zero.partitioned_param_coordinator import iter_params

        # Retrive the parameters that are not available for this module.
        params_to_fetch = frozenset(iter_params(module))
        partitioned_params = []
        for param in params_to_fetch:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                partitioned_params.append(param)

        def _post_backward_function(module, inputs):
            return

        rets = ORTZeROOffloadPreForwardFunction.apply(
            module,
            ds_func_config[1],
            _post_backward_function,
            args_schema,
            kwargs_schema,
            args_tensors,
            kwargs_tensors,
            partitioned_params,
        )

        updated_args_tensors = rets[: len(args_tensors)]
        updated_kwargs_tensors = rets[len(args_tensors) : len(args_tensors) + len(kwargs_tensors)]

        updated_args = unflatten_data_using_schema(updated_args_tensors, args_schema)
        updated_kwargs = unflatten_data_using_schema(updated_kwargs_tensors, kwargs_schema)

        ds_func_config2 = ds_func_store["_post_backward_module_hook"]
        if ds_func_config2[2] is False:
            # initialize the hook func.
            c = ds_func_config2[1]
            cell = _create_closure_for_ds_hook_function(ds_func_config2[0])
            ds_func_config2[1] = FunctionType(c, globals(), c.co_name, None, (cell,))
            ds_func_config2[2] = True

        # _post_backward_module_hook can be traced correctly so we don't need wrap with PythonOp.
        updated_args = ds_func_config2[1](module, updated_args)

        return updated_args, updated_kwargs

    def post_forward_module_apply_impl(
        self,
        run_rtx: _RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,  # ?? can be tensor?
        outputs: ORTModelInputOutputType,  # ?? can be tensor?
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        ds_func_config = ds_func_store["_post_forward_module_hook"]
        if ds_func_config[2] is False:
            # initialize the hook func.
            c = ds_func_config[1]
            cell = _create_closure_for_ds_hook_function(ds_func_config[0])
            ds_func_config[1] = FunctionType(c, globals(), c.co_name, None, (cell,))
            ds_func_config[2] = True

        outputs_tensors, outputs_schema = extract_data_and_schema(outputs)

        def _pre_backward_function(module, inputs, outputs):
            return

        updated_outputs_tensors = ORTZeROOffloadPostForwardFunction.apply(
            module, ds_func_config[1], _pre_backward_function, outputs_schema, *outputs_tensors
        )

        assert len(updated_outputs_tensors) == len(outputs_tensors)

        ds_func_config2 = ds_func_store["_pre_backward_module_hook"]
        if ds_func_config2[2] is False:
            # initialize the hook func.
            c = ds_func_config2[1]
            cell = _create_closure_for_ds_hook_function(ds_func_config2[0])
            ds_func_config2[1] = FunctionType(c, globals(), c.co_name, None, (cell,))
            ds_func_config2[2] = True

        # WARN: we assume updated_output_tensors can REUSE the outputs_schema.
        updated_outputs = unflatten_data_using_schema(updated_outputs_tensors, outputs_schema)

        # WARN: _pre_backward_module_hook's second argument `input is not used, so we just pass a None here.
        updated_outputs = ds_func_config2[1](module, None, updated_outputs)

        return args, updated_outputs

    def post_forward_outmost_module_apply_impl(
        self,
        run_rtx: _RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,  # ?? can be tensor?
        outputs: ORTModelInputOutputType,  # ?? can be tensor?
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:
        ds_func_config = ds_func_store["_end_of_forward_hook"]
        if ds_func_config[2] is False:
            # initialize the hook func.
            c = ds_func_config[1]
            cell = _create_closure_for_ds_hook_function(ds_func_config[0])
            ds_func_config[1] = FunctionType(c, globals(), c.co_name, None, (cell,))
            ds_func_config[2] = True

        outputs_tensors, outputs_schema = extract_data_and_schema(outputs)

        def _pre_backward_function(module, inputs, outputs):
            return

        updated_outputs_tensors = ORTZeROOffloadPostForwardFunction.apply(
            module, ds_func_config[1], _pre_backward_function, outputs_schema, *outputs_tensors
        )

        assert len(updated_outputs_tensors) == len(outputs_tensors)
        updated_outputs = unflatten_data_using_schema(updated_outputs_tensors, outputs_schema)
        return args, updated_outputs
