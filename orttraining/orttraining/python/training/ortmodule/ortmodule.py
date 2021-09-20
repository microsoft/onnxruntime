# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._torch_module_factory import TorchModuleFactory
from ._torch_module_pytorch import TorchModulePytorch
from ._torch_module_ort import TorchModuleORT
from ._custom_op_symbolic_registry import CustomOpSymbolicRegistry
from ._custom_gradient_registry import CustomGradientRegistry
from . import _utils
from .debug_options import DebugOptions
from ._fallback import _FallbackManager, _FallbackPolicy, ORTModuleFallbackException, ORTModuleTorchModelException, wrap_exception
from . import _FALLBACK_INIT_EXCEPTION, MINIMUM_RUNTIME_PYTORCH_VERSION_STR, ORTMODULE_FALLBACK_POLICY, ORTMODULE_FALLBACK_RETRY
from onnxruntime.tools import pytorch_export_contrib_ops

import functools
import torch
from typing import Iterator, Optional, Tuple, TypeVar, Set, Callable
import warnings

# Needed to override PyTorch methods
T = TypeVar('T', bound='Module')


class ORTModule(torch.nn.Module):
    """Extends user's :class:`torch.nn.Module` model to leverage ONNX Runtime super fast training engine.

    ORTModule specializes the user's :class:`torch.nn.Module` model, providing :meth:`~torch.nn.Module.forward`,
    :meth:`~torch.nn.Module.backward` along with all others :class:`torch.nn.Module`'s APIs.

    Args:
        module (torch.nn.Module): User's PyTorch module that ORTModule specializes
        debug_options (:obj:`DebugOptions`, optional): debugging options for ORTModule.
    """

    def __init__(self, module, debug_options=None):

        # NOTE: torch.nn.Modules that call setattr on their internal attributes regularly
        #       (for example PyTorch Lightning), will trigger regular re-exports. This is
        #       because ORTModule auto detects such setattrs on the original module and
        #       marks the model as stale. This is a known limitation. To disable repeated
        #       re-export checks when not required, please set the environment variable
        #       ORTMODULE_SKIPCHECK_POLICY to SKIP_CHECK_BUILD_GRADIENT|SKIP_CHECK_EXECUTION_AGENT

        # Set _is_initialized attribute first which starts off as False.
        # This variable will be used for comparing strings in __setattr__ and __getattr__
        # NOTE: Do not rename/move.
        self._is_initialized = False
        # Python default arguments are evaluated on function definition
        # and not on function invocation. So, if debug_options is not provided,
        # instantiate it inside the function.
        if not debug_options:
            debug_options = DebugOptions()

        # Fallback settings
        self._fallback_manager = _FallbackManager(policy=ORTMODULE_FALLBACK_POLICY,
                                                  retry=ORTMODULE_FALLBACK_RETRY)

        try:
            # Read ORTModule module initialization status
            global _FALLBACK_INIT_EXCEPTION
            if _FALLBACK_INIT_EXCEPTION:
                raise _FALLBACK_INIT_EXCEPTION

            super(ORTModule, self).__init__()

            self._torch_module = TorchModuleFactory()(module, debug_options, self._fallback_manager)

            # Create forward dynamically, so each ORTModule instance will have its own copy.
            # This is needed to be able to copy the forward signatures from the original PyTorch models
            # and possibly have different signatures for different instances.
            def _forward(self, *inputs, **kwargs):
                '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

                ONNX model is exported the first time this method is executed.
                Next, we build a full training graph with module_gradient_graph_builder.
                Finally, we instantiate the ONNX Runtime InferenceSession.
                '''

                return self._torch_module.forward(*inputs, **kwargs)

            # Bind the forward method.
            self.forward = _forward.__get__(self)
            # Copy the forward signature from the _torch_module's forward signature.
            functools.update_wrapper(
                self.forward.__func__, self._torch_module.forward.__func__)

            # Support contrib OPs
            pytorch_export_contrib_ops.register()
            CustomOpSymbolicRegistry.register_all()
            CustomGradientRegistry.register_all()

            # Warn user if there are name collisions between user model's and ORTModule attributes
            # And if there are custom methods defined on the user's model, copy and bind them to
            # ORTModule.
            _utils.check_for_name_collisions_and_bind_methods_to_ortmodule(self, module)

        except ORTModuleFallbackException as e:
            self._torch_module = TorchModulePytorch(module)
            # TODO: Rework by implementing the "__getattribute__" method.
            #       Assigning all default attributes from user's original torch.nn.Module into ORTModule
            self._backward_hooks = module._backward_hooks
            self._forward_hooks = module._forward_hooks
            self._forward_pre_hooks = module._forward_pre_hooks
            self._parameters = module._parameters
            self._buffers = module._buffers
            self._non_persistent_buffers_set = module._non_persistent_buffers_set
            self._is_full_backward_hook = module._is_full_backward_hook
            self._state_dict_hooks = module._state_dict_hooks
            self._load_state_dict_pre_hooks = module._load_state_dict_pre_hooks
            self._modules = module._modules
            self.forward = module.forward

            # Exceptions subject to fallback are handled here
            self._fallback_manager.handle_exception(exception=e,
                                                    log_level=debug_options.logging.log_level)
        except Exception as e:
            self._torch_module = TorchModulePytorch(module)
            # Catch-all FALLBACK_FORCE_TORCH_FORWARD fallback is handled here
            self._fallback_manager.handle_exception(exception=e,
                                                    log_level=debug_options.logging.log_level,
                                                    override_policy=_FallbackPolicy.FALLBACK_FORCE_TORCH_FORWARD)

        # Finally, ORTModule initialization is complete.
        # Assign self._is_initialized to True after all the ORTModule class attributes have been assigned
        # else, they will be assigned to self._torch_module.original_module instead.
        self._is_initialized = True

    # IMPORTANT: DO NOT add code here
    # This declaration is for automatic document generation purposes only
    # The actual forward implementation is bound during ORTModule initialization
    def forward(self, *inputs, **kwargs):
        '''Delegate the :meth:`~torch.nn.Module.forward` pass of PyTorch training to
        ONNX Runtime.

        The first call to forward performs setup and checking steps. During this call,
        ORTModule determines whether the module can be trained with ONNX Runtime. For
        this reason, the first forward call execution takes longer than subsequent calls.
        Execution is interupted if ONNX Runtime cannot process the model for training.

        Args:
            *inputs and **kwargs represent the positional, variable positional, keyword
            and variable keyword arguments defined in the user's PyTorch module's forward
            method. Values can be torch tensors and primitive types.

        Returns:
            The output as expected from the forward method defined by the user's
            PyTorch module. Output values supported include tensors, nested sequences
            of tensors and nested dictionaries of tensor values.
        '''

    def _replicate_for_data_parallel(self):
        """Raises a NotImplementedError exception since ORTModule is not compatible with torch.nn.DataParallel

        torch.nn.DataParallel requires the model to be replicated across multiple devices, and
        in this process, ORTModule tries to export the model to onnx on multiple devices with the same
        sample input. Because of this multiple device export with the same sample input, torch throws an
        exception that reads: "RuntimeError: Input, output and indices must be on the current device"
        which can be vague to the user since they might not be aware of what happens behind the scene.

        We therefore try to preemptively catch use of ORTModule with torch.nn.DataParallel and throw a
        more meaningful exception.

        Users must use torch.nn.parallel.DistributedDataParallel instead of torch.nn.DataParallel
        which does not need model replication and is also recommended by torch to use instead.
        """

        return self._torch_module._replicate_for_data_parallel()

    def add_module(self, name: str, module: Optional['Module']) -> None:
        """Raises a ORTModuleTorchModelException exception since ORTModule does not support adding modules to it"""

        self._torch_module.add_module(name, module)

    @property
    def module(self):
        """The original `torch.nn.Module` that this module wraps.

        This property provides access to methods and properties on the original module.
        """

        # HuggingFace Trainer `save_model` method checks to see if the input model is a HuggingFace PreTrainedModel
        # or if the model has an attribute called `module` which references a HuggingFace PreTrainedModel to save
        # the entire context of the model so that it can be loaded using HuggingFace `from_pretrained` method.
        # This `module` property enables HuggingFace Trainer to retrieve the underlying PreTrainedModel inside ORTModule
        # to save and load a complete checkpoint

        return self._torch_module.module

    ################################################################################
    # The methods below are part of torch.nn.Module API that are encapsulated through
    # TorchModuleInterface
    ################################################################################

    def _apply(self, fn):
        """Override original method to delegate execution to the flattened PyTorch user module"""

        self._torch_module._apply(fn)
        return self

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        """Override :meth:`~torch.nn.Module.apply` to delegate execution to ONNX Runtime"""

        self._torch_module.apply(fn)
        return self

    def _is_training(self):
        return self._torch_module.is_training()

    def train(self: T, mode: bool = True) -> T:
        """Override :meth:`~torch.nn.Module.train` to delegate execution to ONNX Runtime"""

        self.training = mode
        # In a torch.nn.Module, _modules stores all dependent modules (sub-modules) of the current module.
        # in a list so that torch.nn.Module can apply any changes to all sub-modules recursively.
        # Although the _flattened_module and _original_module are dependent modules for ORTModule,
        # they do not show up in _modules because they are abstracted away behind another class,
        # TorchModule. In order to apply changes to those sub-modules, delegate the task to _torch_module
        # which will recursively update the flattened_module and the original module.
        self._torch_module.train(mode)
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override :meth:`~torch.nn.Module.state_dict` to delegate execution to ONNX Runtime"""

        return self._torch_module.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        """Override :meth:`~torch.nn.Module.load_state_dict` to delegate execution to ONNX Runtime"""

        return self._torch_module.load_state_dict(state_dict, strict=strict)

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        """Override :meth:`~torch.nn.Module.register_buffer`"""

        self._torch_module.register_buffer(name, tensor, persistent=persistent)

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        """Override :meth:`~torch.nn.Module.register_parameter`"""

        self._torch_module.register_parameter(name, param)

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        """Override :meth:`~torch.nn.Module.get_parameter`"""

        return self._torch_module.get_parameter(target)

    def get_buffer(self, target: str) -> torch.Tensor:
        """Override :meth:`~torch.nn.Module.get_buffer`"""

        return self._torch_module.get_buffer(target)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Override :meth:`~torch.nn.Module.parameters`"""

        yield from self._torch_module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Override :meth:`~torch.nn.Module.named_parameters`"""

        yield from self._torch_module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        """Override :meth:`~torch.nn.Module.buffers`"""

        yield from self._torch_module.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """Override :meth:`~torch.nn.Module.named_buffers`"""

        yield from self._torch_module.named_buffers(prefix=prefix, recurse=recurse)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override original method to delegate execution to the original PyTorch user module"""

        self._torch_module._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                 missing_keys, unexpected_keys, error_msgs)

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """Override :meth:`~torch.nn.Module.named_children`"""

        yield from self._torch_module.named_children()

    def modules(self) -> Iterator['Module']:
        """Override :meth:`~torch.nn.Module.modules`"""

        yield from self._torch_module.modules()

    def named_modules(self, *args, **kwargs):
        """Override :meth:`~torch.nn.Module.named_modules`"""

        yield from self._torch_module.named_modules(*args, **kwargs)

    def __getattr__(self, name: str):
        if '_is_initialized' in self.__dict__ and self.__dict__['_is_initialized'] == True:
            # If ORTModule is intitialized and attribute is not found in ORTModule,
            # it must be present in the user's torch.nn.Module. Forward the call to
            # the user's model.
            assert '_torch_module' in self.__dict__, "ORTModule does not have a reference to the user's model"
            return getattr(self.module, name)
        else:
            return super(ORTModule, self).__getattr__(name)

    def __setattr__(self, name: str, value) -> None:

        if name in self.__dict__:
            # If the name is an attribute of ORTModule, update only ORTModule
            self.__dict__[name] = value

        elif '_is_initialized' in self.__dict__ and self.__dict__['_is_initialized'] == True:

            assert '_torch_module' in self.__dict__, "ORTModule does not have a reference to the user's model"

            # If the name is an attribute of user model, or is a new attribute, update there.
            # Set the attribute on the user's original module
            setattr(self.module, name, value)
            # Signal to execution manager to re-export the model.
            # Re-export will be avoided if _skip_check is enabled.
            if isinstance(self._torch_module, TorchModuleORT):
                for training_mode in [False, True]:
                    self._torch_module._execution_manager(training_mode).signal_model_changed()

        else:
            # Setting any new attributes should be done on ORTModule only when 'torch_module' is not defined
            self.__dict__[name] = value
