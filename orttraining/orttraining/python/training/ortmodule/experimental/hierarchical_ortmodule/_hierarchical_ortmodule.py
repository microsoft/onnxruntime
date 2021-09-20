# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# debug_options.py
import tempfile
import torch
from ... import ORTModule


class HierarchicalORTModule(torch.nn.Module):
    def __init__(self, module):
        super(HierarchicalORTModule, self).__init__()
        self._initialized = False
        self._original_module = module

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
        # The function "record_args" records the inputs for each nn.Module,
        # and later we will try exporting those nn.Module's with their recorded
        # inputs.
        def recursive_hook(module):
            handle_pool.append(module.register_forward_pre_hook(record_args))
            for name, sub in module._modules.items():
                if isinstance(sub, torch.nn.ModuleList):
                    for name1, sub1 in sub._modules.items():
                        recursive_hook(sub1)
                else:
                    recursive_hook(sub)

        exportable_list = {}

        # Fill "exportable_list". exportable_list[module] = True means
        # "module" can be wrapped as ORTModule. Otherwise, "module" is
        # not exportable to ONNX.
        def check_exportable(module):
            sub_dict = module._modules
            if not sub_dict:
                # No sub-module exists, so this module is a leaf
                # module in overall model hierarchy.
                exportable = True
                # Check if this leaf module is exportable.
                for args in module_arg_pool[module]:
                    try:
                        with tempfile.NamedTemporaryFile(prefix='sub-module') as temp:
                            torch.onnx.export(
                                module, args, temp, opset_version=9)
                    except Exception as e:
                        exportable = False

                exportable_list[module] = exportable
                return exportable
            else:
                sub_exportable = True
                for name, sub_module in sub_dict.items():
                    if isinstance(sub_module, torch.nn.ModuleList):
                        for name1, sub_module1 in sub_module._modules.items():
                            sub_exportable1 = check_exportable(sub_module1)
                            sub_exportable = sub_exportable and sub_exportable1
                    else:
                        sub_exportable1 = check_exportable(sub_module)
                        sub_exportable = sub_exportable and sub_exportable1

                if sub_exportable is False:
                    # At least one existing sub-module is not exportable,
                    # so is the entire module.
                    exportable_list[module] = sub_exportable
                    return sub_exportable
                else:
                    # Now, we know all sub-modules are exportable, so
                    # we are going to check if the composition of them
                    # is still exportable at this module level.
                    module_exportable = True
                    for args in module_arg_pool[module]:
                        try:
                            with tempfile.NamedTemporaryFile(prefix='sub-module') as temp:
                                torch.onnx.export(
                                    module, args, temp, opset_version=9)
                        except Exception as e:
                            # If this module is not exportable for one arg
                            # group, we say this module is not exportable.
                            module_exportable = False
                            # Already found a broken case.
                            # No need to check next case.
                            break

                    exportable_list[module] = module_exportable
                    return exportable_list[module]

        # Top-down wrapper to replace nn.Module's with ORTModule.
        # Note that using bottom-up wrapper may lead to much
        # ORTModule instances and each ORTModule owns a much smaller graph.
        def recursive_wrap(module):
            sub_dict = module._modules
            for name, sub in sub_dict.items():
                if isinstance(sub, torch.nn.ModuleList):
                    # We encounter a list of sub-modules.
                    # Let's wrap them one-by-one.
                    for name1, sub1 in sub._modules.items():
                        if is_supported(sub1):
                            sub._modules[name1] = ORTModule(sub1)
                        else:
                            recursive_wrap(sub1)
                else:
                    if is_supported(sub):
                        # Just wrap it as ORTModule when possible.
                        sub_dict[name] = ORTModule(sub)
                    else:
                        # This sub-module is not exportable to ONNX
                        # Let's check its sub-modules.
                        recursive_wrap(sub)

        # Add a hook to record forward's input for all modules.
        recursive_hook(self._original_module)

        # Run forward with actual input to record all possible
        # inputs for all invoked modules.
        y = self._original_module(*args, **kwargs)

        # We already have "supported_modules" so
        # we no longer need those hooks in forward pass.
        for h in handle_pool:
            h.remove()

        # Try exporter on all module-input pairs. If a module can be exported with
        # all its recorded inputs, then it's exporable.
        check_exportable(self._original_module)

        # A naive way of determining if ORT can run nn.Module
        def is_supported(module):
            # This is what Yuan called GetCapability().
            return module in exportable_list and exportable_list[module]

        # Recursively wrap nn.Module as ORTModule so supported computation
        # would be delegate to ORT and the rest may happen in Pytorch.
        def recursive_wrap(module):
            sub_dict = module._modules
            for name, sub in sub_dict.items():
                if isinstance(sub, torch.nn.ModuleList):
                    for name1, sub1 in sub._modules.items():
                        if is_supported(sub1):
                            sub._modules[name1] = ORTModule(sub1)
                        else:
                            recursive_wrap(sub1)
                else:
                    if is_supported(sub):
                        sub_dict[name] = ORTModule(sub)
                    else:
                        recursive_wrap(sub)

        recursive_wrap(self._original_module)

    def forward(self, *inputs, **kwargs):
        if not self._initialized:
            self._initialized = True
            self._initialize(*inputs, **kwargs)
        return self._original_module.forward(*inputs, **kwargs)
