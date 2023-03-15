# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import warnings
from collections import abc
import torch


class _ModuleHookSubscriberBase:
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

    def __init__(self, start_step=None, end_step=None):
        self._start_step = start_step
        self._end_step = end_step

    # subscriber.forward(ctx, val, depth, name, ctx.current_step)
    def forward(self, activation_sequence_id, activation, depth, name, step):
        raise NotImplementedError()

    def backward(self, activation_sequence_id, activation, depth, name, step):
        raise NotImplementedError()


# Used to track current execution step, e.g. how many forward/backward path is called.
_EXECUTION_GLOBAL_STEP = 0
# Used to store the activation tensor names, if already handled, then skipped.
# Need to clear after each step.
_OBSERVED_ACTIVATION_NAMES = {}
# Used to store the depth of each module, which indicate the indentation level of the module.
_MODULE_INDEX_TO_DEPTH = {}
# Used to store the unique id of each sequential activation.
_MODULE_TO_MODULE_INDEX = {}


class _InspectActivation(torch.autograd.Function):
    """
    This class is used to run the subscriber's forward and backward function.
    It also manages step related checks, including run subscribers only for between start_step and end_step.
    """

    @staticmethod
    def forward(ctx, name, id, depth, input, subscriber):
        global _EXECUTION_GLOBAL_STEP
        val = None
        if input is None or not isinstance(input, torch.Tensor):
            val = input
        else:
            val = input.detach().clone()
        ctx.current_step = _EXECUTION_GLOBAL_STEP
        if subscriber._start_step is not None and subscriber._end_step is not None:
            if ctx.current_step >= subscriber._start_step and ctx.current_step < subscriber._end_step:
                subscriber.forward(ctx, val, depth, name, ctx.current_step)

        ctx.name = name
        ctx.id = id
        ctx.depth = depth
        ctx.subscriber = subscriber

        return input.detach() if input is not None else None

    @staticmethod
    def backward(ctx, grad_output):
        global _OBSERVED_ACTIVATION_NAMES
        val = None
        if grad_output is None or not isinstance(grad_output, torch.Tensor):
            val = grad_output
        else:
            val = grad_output.detach().clone()

        if ctx.subscriber._start_step is not None and ctx.subscriber._end_step is not None:
            if ctx.current_step >= ctx.subscriber._start_step and ctx.current_step < ctx.subscriber._end_step:
                ctx.subscriber.backward(ctx, val, ctx.depth, ctx.name, ctx.current_step)

        return None, None, None, grad_output.detach() if grad_output is not None else None, None


class _IncrementStep(torch.autograd.Function):
    """
    This class is used to manage the global execution step, e.g.
    global step increment by one, once a full forward path completed and state clear.

    This autograd Function is registered as a post forward hook to the root module. This is only
    expected to run once for the first tensor output, otherwise, the logic is not right.
    """

    @staticmethod
    def forward(ctx, input):
        global _EXECUTION_GLOBAL_STEP, _OBSERVED_ACTIVATION_NAMES

        print(f"{'='*6} Completed forward pass for STEP {_EXECUTION_GLOBAL_STEP} {'='*6}")
        _EXECUTION_GLOBAL_STEP += 1
        _OBSERVED_ACTIVATION_NAMES = {}

        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SubscriberManager(object):
    """
    This class is used to manage all the subscribers, and register post forward hook to the root module.
    _register_subscriber() is used to register a subscriber, and can be chained with other register calls.

    Currently, the hook handled here is post forward hook for nn.Module. Hook is registered for all nn.Modules
    recursively. Each hook insert a PythonOp for every tensor output generated by the corresponding module.
    Each subscriber implementation is called in the PythonOp's forward function, and backward function.

    There is one special handling for global step increment and state clear. A post forward hook is registered
    for the outside-most module, which is the root module. In that hook, _IncrementStep is called, which will
    increase the step by 1 (check _IncrementStep for details).
    """

    registered_subscribers = set()

    def __init__(self):
        pass

    @staticmethod
    def subscribe(module, subscribers):
        if not isinstance(module, torch.nn.Module):
            raise ValueError("module must be a torch.nn.Module instance")

        SubscriberManager.reset_all_states()

        from onnxruntime.training.ortmodule import ORTModule

        if isinstance(module, ORTModule):
            global _EXECUTION_GLOBAL_STEP
            # The first two steps are for preparing to export ONNX, so we use negative number for it.
            _EXECUTION_GLOBAL_STEP -= 2
            module = module.module

        for subscriber in subscribers:
            if not isinstance(subscriber, _ModuleHookSubscriberBase):
                raise ValueError("subscriber must be a _ModuleHookSubscriberBase instance")
            SubscriberManager.registered_subscribers.add(subscriber)

        SubscriberManager._initialize(module)

    @staticmethod
    def reset_all_states():
        global _EXECUTION_GLOBAL_STEP, _OBSERVED_ACTIVATION_NAMES, _MODULE_INDEX_TO_DEPTH, _MODULE_TO_MODULE_INDEX
        _EXECUTION_GLOBAL_STEP = 0
        _OBSERVED_ACTIVATION_NAMES = {}
        _MODULE_INDEX_TO_DEPTH = {}
        _MODULE_TO_MODULE_INDEX = {}

    @staticmethod
    def _initialize(module):
        if len(SubscriberManager.registered_subscribers) == 0:
            warnings.warn("No subscribers are registered, skip insert hooks.")
            return

        count = [0]
        # Register post forward hook for every module, inside the hook, we loop every tensor output of the module,
        # and wrap it with a autograd Function called _InspectActivation (which take in a tensor and return the same
        # tensor). In this way, we keep ORT and PyTorch run have the same boundary to check activation equality.
        SubscriberManager._register_hooks_recursively(module, 1, count)

        # Register post forward hook for outmost module, then we increase the dump step.
        # Be noted, if backward is not triggered, the global dump step remain the original number,
        # which means the subsequent run will override the previous dump files. This indeed happens imagine ORTModule
        # firstly export graph (run the forward only), after gradient graph is built, another forward+backward is
        # triggered, override the previous dump files.
        def _post_forward_outmost_module_hook(module, inputs, outputs2):
            def _apply_to_tensors_func(index, outputs):
                return _IncrementStep.apply(outputs)

            return SubscriberManager._apply_to_tensors(module, 0, outputs2, _apply_to_tensors_func, True)

        module.register_forward_hook(_post_forward_outmost_module_hook)

    @staticmethod
    def _register_hooks_recursively(module, depth, count):
        global _MODULE_TO_MODULE_INDEX, _MODULE_INDEX_TO_DEPTH
        id = count[0]
        _MODULE_INDEX_TO_DEPTH[id] = depth
        _MODULE_TO_MODULE_INDEX[module] = id

        for child in module.children():
            if isinstance(child, torch.nn.Module) and child not in _MODULE_TO_MODULE_INDEX:
                count[0] += 1
                SubscriberManager._register_hooks_recursively(child, depth + 1, count)

        module.register_forward_hook(SubscriberManager._post_forward_module_hook)

    @staticmethod
    def _post_forward_module_hook(module, input, output):
        global _MODULE_TO_MODULE_INDEX
        global _MODULE_INDEX_TO_DEPTH
        global _OBSERVED_ACTIVATION_NAMES

        if module in _MODULE_TO_MODULE_INDEX and isinstance(module, torch.nn.Module):
            id = _MODULE_TO_MODULE_INDEX[module]

            def _apply_to_tensors_func(index, activation_tensor):
                global _OBSERVED_ACTIVATION_NAMES
                name = f"{module.__class__.__name__}_{id}_{index}th_output"
                if name not in _OBSERVED_ACTIVATION_NAMES:
                    _OBSERVED_ACTIVATION_NAMES[name] = True
                    output = activation_tensor
                    for subscriber in SubscriberManager.registered_subscribers:
                        output = _InspectActivation.apply(name, id, _MODULE_INDEX_TO_DEPTH[id], output, subscriber)
                    return output
                else:
                    return activation_tensor

            return SubscriberManager._apply_to_tensors(module, id, output, _apply_to_tensors_func)

    @staticmethod
    def _is_builtin_type(obj):
        # https://stackoverflow.com/a/17795199
        return obj.__class__.__module__ == "__builtin__" or obj.__class__.__module__ == "builtins"

    @staticmethod
    def _apply_to_tensors(module, id, tensors, func, handle_first_tensor_only=False):
        """
        Apply func to all tensors in the given object.
        module: the module that generates the tensors.
        id: the unique id of the module.
        tensors: the object that contains activation tensors.
        func: the function to apply to the tensors.
        """
        idx = [0]

        def flatten_outputs(outputs, id, index, func):
            if isinstance(outputs, abc.Sequence):
                touched_outputs = []
                for output in outputs:
                    touched_output = SubscriberManager._apply_to_tensors(
                        module, id, output, func, handle_first_tensor_only
                    )
                    touched_outputs.append(touched_output)
                return outputs.__class__(touched_outputs)
            elif isinstance(outputs, abc.Mapping):
                # apply inplace to avoid recreating dict inherited objects
                for key, value in outputs.items():
                    outputs[key] = SubscriberManager._apply_to_tensors(
                        module, id, outputs[key], func, handle_first_tensor_only
                    )
                return outputs

            elif type(outputs) is torch.Tensor:
                cur_id = index[0]
                index[0] += 1
                if handle_first_tensor_only is True and cur_id > 0:
                    return outputs
                else:
                    return func(cur_id, outputs)

            else:
                if not SubscriberManager._is_builtin_type(outputs):
                    raise RuntimeError(f"Unknown type {type(outputs)}")
                return outputs

        return flatten_outputs(tensors, id, idx, func)
