# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# isort: skip_file
# Import ordering is important in this module to aviod circular dependencies
import logging
from ._torch_module_factory import TorchModuleFactory
from ._torch_module_ort import TorchModuleORT
from ._custom_op_symbolic_registry import CustomOpSymbolicRegistry
from ._custom_gradient_registry import CustomGradientRegistry
from . import _utils
from .options import DebugOptions
from ._fallback import _FallbackManager, _FallbackPolicy, ORTModuleFallbackException
from ._logger import ortmodule_loglevel_to_python_loglevel
from onnxruntime.training import ortmodule

from onnxruntime.tools import pytorch_export_contrib_ops

import torch
from typing import Iterator, Optional, OrderedDict, Tuple, TypeVar, Callable

# Needed to override PyTorch methods
T = TypeVar("T", bound="torch.nn.Module")


class ORTModule(torch.nn.Module):
    """Extends user's :class:`torch.nn.Module` model to leverage ONNX Runtime super fast training engine.

    ORTModule specializes the user's :class:`torch.nn.Module` model, providing :meth:`~torch.nn.Module.forward`,
    :meth:`~torch.nn.Module.backward` along with all others :class:`torch.nn.Module`'s APIs.

    Args:
        module (torch.nn.Module): User's PyTorch module that ORTModule specializes
        debug_options (:obj:`DebugOptions`, optional): debugging options for ORTModule.
    """

    def __init__(self, module: torch.nn.Module, debug_options: Optional[DebugOptions] = None):
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

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(ortmodule_loglevel_to_python_loglevel(debug_options.logging.log_level))

        # Fallback settings
        self._fallback_manager = _FallbackManager(
            pytorch_module=module,
            policy=ortmodule.ORTMODULE_FALLBACK_POLICY,
            retry=ortmodule.ORTMODULE_FALLBACK_RETRY,
            logger=self._logger,
        )

        try:
            # Read ORTModule module initialization status
            if ortmodule._FALLBACK_INIT_EXCEPTION:
                raise ortmodule._FALLBACK_INIT_EXCEPTION

            super().__init__()

            self._torch_module = TorchModuleFactory()(module, debug_options, self._fallback_manager, self._logger)

            _utils.patch_ortmodule_forward_method(self)

            # Support contrib OPs
            pytorch_export_contrib_ops.register()
            CustomOpSymbolicRegistry.register_all(
                self._torch_module._execution_manager(module.training)._runtime_options.onnx_opset_version
            )
            CustomGradientRegistry.register_all()

            # Warn user if there are name collisions between user model's and ORTModule attributes
            # And if there are custom methods defined on the user's model, copy and bind them to
            # ORTModule.
            _utils.check_for_name_collisions_and_bind_methods_to_ortmodule(self, module, self._logger)

        except ORTModuleFallbackException as e:
            # Although backend is switched to PyTorch here,
            # it is up to _FallbackManager to actually terminate execution or fallback
            _utils.switch_backend_to_pytorch(self, module)

            # Exceptions subject to fallback are handled here
            self._fallback_manager.handle_exception(exception=e, log_level=debug_options.logging.log_level)
        except Exception as e:
            # Although backend is switched to PyTorch here,
            # it is up to _FallbackManager to actually terminate execution or fallback
            _utils.switch_backend_to_pytorch(self, module)

            # Catch-all FALLBACK_FORCE_TORCH_FORWARD fallback is handled here
            self._fallback_manager.handle_exception(
                exception=e,
                log_level=debug_options.logging.log_level,
                override_policy=_FallbackPolicy.FALLBACK_FORCE_TORCH_FORWARD,
            )

        self.train(module.training)
        # Finally, ORTModule initialization is complete.
        # Assign self._is_initialized to True after all the ORTModule class attributes have been assigned
        # else, they will be assigned to self._torch_module.original_module instead.
        self._is_initialized = True

        # del the ort._modules so that all reference to  ort._modules will be forward to the underlying torch_model
        # through '__getattr__'
        del self._modules

    # IMPORTANT: DO NOT add code here
    # This declaration is for automatic document generation purposes only
    # The actual forward implementation is bound during ORTModule initialization
    def forward(self, *inputs, **kwargs):
        """Delegate the :meth:`~torch.nn.Module.forward` pass of PyTorch training to ONNX Runtime.

        The first call to forward performs setup and checking steps. During this call,
        ORTModule determines whether the module can be trained with ONNX Runtime. For
        this reason, the first forward call execution takes longer than subsequent calls.
        Execution is interupted if ONNX Runtime cannot process the model for training.

        Args:
            inputs:  positional, variable positional inputs to the PyTorch module's forward method.
            kwargs: keyword and variable keyword arguments to the PyTorch module's forward method.

        Returns:
            The output as expected from the forward method defined by the user's
            PyTorch module. Output values supported include tensors, nested sequences
            of tensors and nested dictionaries of tensor values.
        """

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

    def add_module(self, name: str, module: Optional[torch.nn.Module]) -> None:
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

    def apply(self: T, fn: Callable[[torch.nn.Module], None]) -> T:
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

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Override :meth:`~torch.nn.Module.state_dict` to delegate execution to ONNX Runtime"""

        return self._torch_module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: "OrderedDict[str, torch.Tensor]", strict: bool = True):
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

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Override :meth:`~torch.nn.Module.named_parameters`"""

        yield from self._torch_module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        """Override :meth:`~torch.nn.Module.buffers`"""

        yield from self._torch_module.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """Override :meth:`~torch.nn.Module.named_buffers`"""

        yield from self._torch_module.named_buffers(prefix=prefix, recurse=recurse)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Override original method to delegate execution to the original PyTorch user module"""

        self._torch_module._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def named_children(self) -> Iterator[Tuple[str, torch.nn.Module]]:
        """Override :meth:`~torch.nn.Module.named_children`"""

        yield from self._torch_module.named_children()

    def modules(self) -> Iterator[torch.nn.Module]:
        """Override :meth:`~torch.nn.Module.modules`"""

        yield from self._torch_module.modules()

    def named_modules(self, *args, **kwargs):
        """Override :meth:`~torch.nn.Module.named_modules`"""

        yield from self._torch_module.named_modules(*args, **kwargs)

    def __getattr__(self, name: str):
        if "_is_initialized" in self.__dict__ and self.__dict__["_is_initialized"] is True:
            # If ORTModule is initialized and attribute is not found in ORTModule,
            # it must be present in the user's torch.nn.Module. Forward the call to
            # the user's model.
            assert "_torch_module" in self.__dict__, "ORTModule does not have a reference to the user's model"
            return getattr(self.module, name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.__dict__:
            # If the name is an attribute of ORTModule, update only ORTModule
            self.__dict__[name] = value

        elif "_is_initialized" in self.__dict__ and self.__dict__["_is_initialized"] is True:
            assert "_torch_module" in self.__dict__, "ORTModule does not have a reference to the user's model"

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

    def __getstate__(self):
        state = _utils.get_state_after_deletion_of_non_ortmodule_methods(self, self.module)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Re-register contrib OPs
        pytorch_export_contrib_ops.register()
        CustomOpSymbolicRegistry.register_all(
            self._torch_module._execution_manager(self.module.training)._runtime_options.onnx_opset_version
        )
        CustomGradientRegistry.register_all()

        # Re-initialize the ORTModule forward method
        _utils.patch_ortmodule_forward_method(self)

        # Re-bind users custom methods to ORTModule
        _utils.check_for_name_collisions_and_bind_methods_to_ortmodule(self, self.module, self._logger)
