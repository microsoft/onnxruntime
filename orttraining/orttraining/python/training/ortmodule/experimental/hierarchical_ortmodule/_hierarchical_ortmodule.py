# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _hierarchical_ortmodule.py
import tempfile
import warnings

import torch

from onnxruntime.training import ortmodule
from onnxruntime.training.ortmodule import ORTModule
from onnxruntime.training.ortmodule.options import DebugOptions, LogLevel

# nn.Module's in this set are considered exportable to ONNX.
# For other nn.Module's, torch.onnx.export is called to check if
# they are exportable.
_force_exportable_set = {torch.nn.Linear, torch.nn.Identity, torch.nn.modules.linear.NonDynamicallyQuantizableLinear}


class _IteratedORTModule(torch.nn.Module):
    """
    It's possible that a module instance is called multiple times in a single forward() call with different inputs.
    If the number of inputs or the data types are different, the exported graph for a given input set cannot be used
    for others. The _IteratedORTModule class is used to handle this case. It creates multiple ORTModule instances
    for a given nn.Module instance and uses one of them for each input set.

    NOTE that we assume that for each step run, the running order of different input sets are same.
    If it's not this case (e.g., a module is used for checkpointing so that a same input set is used twice),
    this class cannot handle it. An ideal way is to maintain a map from different input sets (maybe compute a hash)
    to ORTModule instances.
    """

    def __init__(self, module, count, log_level, save_onnx, onnx_prefix):
        super().__init__()
        assert count > 1
        self._count = count
        self._it = count - 1
        self._ortmodules = []
        for idx in range(count):
            self._ortmodules.append(
                ORTModule(
                    module,
                    debug_options=DebugOptions(
                        log_level=log_level, save_onnx=save_onnx, onnx_prefix=onnx_prefix + "_it" + str(idx)
                    ),
                )
            )

    def forward(self, *inputs, **kwargs):
        self._it = (self._it + 1) % self._count
        return self._ortmodules[self._it](*inputs, **kwargs)


class HierarchicalORTModule(torch.nn.Module):
    """
    Recursively wraps submodules of `module` as ORTModule whenever possible
    Similarly to ORTModule, the actual wrapping happens in its first `forward` call during Pytorch-to-ONNX export.
    Supported computation is delegated to ONNX Runtime and unsupported computation is still done by PyTorch.
    Args:
        module (torch.nn.Module): User's PyTorch module that HierarchicalORTModule specializes.

    Example::

        import torch
        from torch.utils.checkpoint import checkpoint
        from onnxruntime.training.ortmodule.experimental.hierarchical_ortmodule import HierarchicalORTModule


        class Foo(torch.nn.Module):
            def __init__(self):
                super(Foo, self).__init__()
                self.l1 = torch.nn.Linear(2, 2)
                self.l2 = torch.nn.Linear(2, 2)
                self.l3 = torch.nn.Linear(2, 2)

            def forward(self, x):
                def custom():
                    def custom_forward(x_):
                        return self.l2(x_)
                    return custom_forward
                z = self.l1(checkpoint(custom(), self.l3(x)))
                return z

        x = torch.rand(2)
        m = HierarchicalORTModule(Foo())
        y = m(x)

    """

    def __init__(self, module, debug_options=None):
        self._initialized = False
        super().__init__()
        self._original_module = module
        self._log_level = debug_options.logging.log_level if debug_options else LogLevel.ERROR
        self._save_onnx = debug_options.save_onnx_models.save if debug_options else False
        self._name_prefix = debug_options.save_onnx_models.name_prefix if debug_options else ""

    def _initialize(self, *args, **kwargs):
        handle_pool = []
        module_arg_pool = {}

        # A forward pre-hook to record inputs for each nn.Module.
        def record_args(module, args):
            if module in module_arg_pool:
                module_arg_pool[module].append(args)
            else:
                module_arg_pool[module] = [args]

        # Recursively hook "record_args" to module and all its sub-modules.
        # The function "record_args" records the inputs for each nn.Module and later we will try exporting
        # those nn.Module's with their recorded inputs.
        # NOTE that if a module is not called from forward(), it will fail to be captured by this hook.
        def recursive_hook(module):
            # We cannot skip module in allowlist because it's possible that a module is called multiple times
            # so that we still need to know the number of different input sets and use _IteratedORTModule to handle it.
            handle_pool.append(module.register_forward_pre_hook(record_args))
            for sub_module in module._modules.values():
                if isinstance(sub_module, torch.nn.ModuleList):
                    for sub_module_item in sub_module._modules.values():
                        recursive_hook(sub_module_item)
                else:
                    recursive_hook(sub_module)

        exportable_list = {}

        # Fill "exportable_list". exportable_list[module] = True means
        # "module" can be wrapped as ORTModule. Otherwise, "module" is
        # not exportable to ONNX.
        def check_exportable(module):
            def try_export(module, args):
                try:
                    with tempfile.NamedTemporaryFile(prefix="sub-module") as temp, torch.no_grad():
                        torch.onnx.export(
                            module,
                            args,
                            temp,
                            opset_version=ortmodule.ONNX_OPSET_VERSION,
                            do_constant_folding=False,
                            export_params=False,
                            keep_initializers_as_inputs=True,
                            training=torch.onnx.TrainingMode.TRAINING,
                        )
                except Exception as e:
                    if self._log_level <= LogLevel.WARNING:
                        warnings.warn(
                            f"Failed to export module with type {type(module).__name__}. Error message: {e!s}",
                            UserWarning,
                        )
                    return False
                return True

            if type(module) in _force_exportable_set:
                exportable_list[module] = True
                return

            # It's possible that the model runs a module by calling some other function instead of forward()
            # so that the module is not captured by the forward pre-hook. In this case, we will treat it as
            # not exportable for now.
            module_exportable = module in module_arg_pool
            if module_exportable:
                for args in module_arg_pool[module]:
                    if not try_export(module, args):
                        module_exportable = False
                        break
            elif self._log_level <= LogLevel.WARNING:
                warnings.warn(
                    f"Module with type {type(module).__name__} is not exportable because it's not in module_arg_pool.",
                    UserWarning,
                )

            exportable_list[module] = module_exportable
            if module_exportable:
                return

            sub_module_dict = module._modules
            if not sub_module_dict:
                # No sub-module exists, so this module is a leaf
                return

            for sub_module in sub_module_dict.values():
                if isinstance(sub_module, torch.nn.ModuleList):
                    for sub_module_item in sub_module._modules.values():
                        check_exportable(sub_module_item)
                else:
                    check_exportable(sub_module)

        # Add a hook to record forward's input for all modules.
        recursive_hook(self._original_module)

        # Run forward with actual input to record all possible
        # inputs for all invoked modules.
        with torch.no_grad():
            _ = self._original_module(*args, **kwargs)

        # We already have "supported_modules" so
        # we no longer need those hooks in forward pass.
        for handle in handle_pool:
            handle.remove()

        # Try exporter on all module-input pairs. If a module can be exported with
        # all its recorded inputs, then it's exporable.
        check_exportable(self._original_module)

        # A naive way of determining if ORT can run nn.Module
        def is_supported(module):
            return module in exportable_list and exportable_list[module]

        # Top-down wrapper to replace nn.Module's with ORTModule.
        # Note that using bottom-up wrapper may lead to much
        # ORTModule instances and each ORTModule owns a much smaller graph.
        def recursive_wrap(module, save_onnx=False, onnx_prefix=""):
            sub_module_dict = module._modules
            for name, sub_module in sub_module_dict.items():
                new_prefix = onnx_prefix + "_" + name
                if isinstance(sub_module, torch.nn.ModuleList):
                    # We encounter a list of sub-modules.
                    # Let's wrap them one-by-one.
                    idx = 0
                    for item_name, sub_module_item in sub_module._modules.items():
                        # Avoid saving too many graphs.
                        new_save_onnx = save_onnx and idx == 0
                        sub_new_prefix = new_prefix + "_" + item_name
                        if is_supported(sub_module_item):
                            if sub_module_item in module_arg_pool and len(module_arg_pool[sub_module_item]) > 1:
                                sub_module._modules[item_name] = _IteratedORTModule(
                                    sub_module_item,
                                    len(module_arg_pool[sub_module_item]),
                                    self._log_level,
                                    new_save_onnx,
                                    sub_new_prefix,
                                )
                            else:
                                sub_module._modules[item_name] = ORTModule(
                                    sub_module_item,
                                    debug_options=DebugOptions(
                                        log_level=self._log_level, save_onnx=new_save_onnx, onnx_prefix=sub_new_prefix
                                    ),
                                )
                        else:
                            recursive_wrap(sub_module_item, new_save_onnx, sub_new_prefix)
                        idx += 1
                else:
                    if is_supported(sub_module):
                        # Just wrap it as ORTModule when possible.
                        if sub_module in module_arg_pool and len(module_arg_pool[sub_module]) > 1:
                            sub_module_dict[name] = _IteratedORTModule(
                                sub_module, len(module_arg_pool[sub_module]), self._log_level, save_onnx, new_prefix
                            )
                        else:
                            sub_module_dict[name] = ORTModule(
                                sub_module,
                                debug_options=DebugOptions(
                                    log_level=self._log_level, save_onnx=save_onnx, onnx_prefix=new_prefix
                                ),
                            )
                    else:
                        # This sub-module is not exportable to ONNX
                        # Let's check its sub-modules.
                        recursive_wrap(sub_module, save_onnx, new_prefix)

        if is_supported(self._original_module):
            self._original_module = ORTModule(
                self._original_module,
                debug_options=DebugOptions(
                    log_level=self._log_level, save_onnx=self._save_onnx, onnx_prefix=self._name_prefix
                ),
            )
        else:
            recursive_wrap(self._original_module, self._save_onnx, self._name_prefix)
        if self._log_level <= LogLevel.WARNING:
            warnings.warn(
                f"Wrapped module: {self._original_module!s}.",
                UserWarning,
            )
        self._initialized = True

    def forward(self, *inputs, **kwargs):
        if not self._initialized:
            self._initialize(*inputs, **kwargs)
        # forward can be run only after initialization is done.
        assert self._initialized
        return self._original_module(*inputs, **kwargs)
