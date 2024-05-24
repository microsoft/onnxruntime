# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.capi._pybind_state import is_torch_interop_default_on
from onnxruntime.training import ortmodule


class Enabler:
    def __init__(self):
        self._state = False

        # This is to indicate whether custom autograd support has been enabled or not (despite of the current state)
        # in current process.
        self._already_enabled = False

    @property
    def state(self):
        return self._state

    @property
    def already_enabled(self):
        # Once enabled, this flag will be True.
        return self._already_enabled

    @state.setter
    def state(self, val):
        self._state = val
        if self._already_enabled is False and val is True:
            self._already_enabled = True


custom_autograd_function_enabler = Enabler()


# Legacy API to enable the custom autograd, keep its name with default value for compatibility.
def enable_custom_autograd_support(to_enable=True):
    import atexit

    from torch.onnx import register_custom_op_symbolic, unregister_custom_op_symbolic

    from onnxruntime.capi._pybind_state import (
        register_backward_runner,
        register_forward_runner,
        unregister_python_functions,
    )
    from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_interop_utils

    from ._custom_autograd_function_exporter import _export

    if to_enable is True and custom_autograd_function_enabler.state is False:
        if custom_autograd_function_enabler.already_enabled is False:
            # Initialize static objects needed to run custom autograd.Function's.

            register_forward_runner(torch_interop_utils.get_custom_function_forward_runner())
            register_backward_runner(torch_interop_utils.get_custom_function_backward_runner())

            # Unregister all python functions automatically upon normal interpreter termination.
            atexit.register(unregister_python_functions)
            # Clear all gradient functions, to avoid a deadlock issue.
            # Check the called function for more detailed comments.
            atexit.register(torch_interop_utils.clear_all_grad_fns)

        try:
            # This is for the latest Pytorch nightly after this commit:
            # https://github.com/pytorch/pytorch/commit/11bc435622e6b7207bbf37ed1aafe999e1f296ec
            register_custom_op_symbolic("prim::PythonOp", _export, 1)
        except Exception:
            # This applies to Pytorch 1.9 and 1.9.1.
            register_custom_op_symbolic("::prim_PythonOp", _export, 1)

        custom_autograd_function_enabler.state = True
    elif to_enable is False and custom_autograd_function_enabler.state is True:
        # We don't need remove the registered runner because it won't be used if we disable the feature.
        # But we need unregister the PythonOp custom operator function.
        try:
            # This is for the latest Pytorch nightly after this commit:
            # https://github.com/pytorch/pytorch/commit/11bc435622e6b7207bbf37ed1aafe999e1f296ec
            unregister_custom_op_symbolic("prim::PythonOp", 1)
        except Exception:
            # This applies to Pytorch 1.9 and 1.9.1.
            unregister_custom_op_symbolic("::prim_PythonOp", 1)

        custom_autograd_function_enabler.state = False


# Enable the custom autograd by default when PythonOp backend support is enabled during build.
enable_custom_autograd_support(
    ortmodule._defined_from_envvar("ORTMODULE_ENABLE_CUSTOM_AUTOGRAD", 1) and is_torch_interop_default_on()
)
