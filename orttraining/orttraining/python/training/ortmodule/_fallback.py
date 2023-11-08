# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from enum import IntFlag
from logging import Logger
from typing import Optional

import torch

from . import _logger, _utils
from ._fallback_exceptions import wrap_exception  # noqa: F401
from ._fallback_exceptions import (
    ORTModuleDeviceException,
    ORTModuleFallbackException,
    ORTModuleInitException,
    ORTModuleIOError,
    ORTModuleONNXModelException,
    ORTModuleTorchModelException,
)


class _FallbackPolicy(IntFlag):
    """Policy to trigger fallback from ONNX Runtime engine to PyTorch

    Each policy can be combined with the others (using |) in order to aggregate them
    """

    FALLBACK_DISABLE = 1
    FALLBACK_FORCE_TORCH_FORWARD = 2
    FALLBACK_FORCE_TORCH_BACKWARD = 4
    FALLBACK_UNSUPPORTED_DEVICE = 8
    FALLBACK_UNSUPPORTED_DATA = 16
    FALLBACK_UNSUPPORTED_TORCH_MODEL = 32
    FALLBACK_UNSUPPORTED_ONNX_MODEL = 64
    FALLBACK_BAD_INITIALIZATION = 128

    def is_set(self, policy):
        """Check whether `policy` is set on the `_FallbackPolicy` instance

        FALLBACK_DISABLE implies the check will always fail and return False
        """

        return not self.is_disabled() and policy in self

    def is_disabled(self):
        """Check whether `_FallbackPolicy.FALLBACK_DEVICE` is set on the `_FallbackPolicy` instance"""

        return _FallbackPolicy.FALLBACK_DISABLE in self


class _FallbackManager:
    """Manages fallbacks based on incoming exceptions and specified policies

    The basic algorithm is based on a dictionary whose keys are the supported fallback policies
    and and values are a set of Exception that must be detected.

    When an exception that matches one of the enabled policies are detected,
    a fallback will be pending to execute by ORTModule frontend. If `retry` is False,
    ORTModule will fallback to PyTorch on the following steps. Otherwise, ORTModule [re]try
    to run the model using ORT backend in every step


    On the other hand, when the exception doesn't match any enabled policy, the exception will
    be raised to the user, terminating execution
    """

    def __init__(self, pytorch_module: torch.nn.Module, policy: _FallbackPolicy, retry: bool, logger: Logger):
        self._original_module = pytorch_module

        # Read policy from environment variable for testing purposes
        policy = os.getenv("ORTMODULE_FALLBACK_POLICY", policy)
        if isinstance(policy, str):
            policy = _FallbackPolicy[policy]

        # Read retry from environment variable for testing purposes
        retry = os.getenv("ORTMODULE_FALLBACK_RETRY", str(retry)).lower() in ["true", "1", "yes"]

        self._policy_exception_map = {
            _FallbackPolicy.FALLBACK_FORCE_TORCH_FORWARD.value: {
                ORTModuleFallbackException,
                ORTModuleDeviceException,
                ORTModuleIOError,
                ORTModuleTorchModelException,
                ORTModuleONNXModelException,
            },
            _FallbackPolicy.FALLBACK_FORCE_TORCH_BACKWARD.value: {
                ORTModuleFallbackException,
                ORTModuleDeviceException,
                ORTModuleIOError,
                ORTModuleTorchModelException,
                ORTModuleONNXModelException,
            },
            _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE.value: {ORTModuleDeviceException},
            _FallbackPolicy.FALLBACK_UNSUPPORTED_DATA.value: {ORTModuleIOError},
            _FallbackPolicy.FALLBACK_UNSUPPORTED_TORCH_MODEL.value: {ORTModuleTorchModelException},
            _FallbackPolicy.FALLBACK_UNSUPPORTED_ONNX_MODEL.value: {ORTModuleONNXModelException},
            _FallbackPolicy.FALLBACK_BAD_INITIALIZATION.value: {ORTModuleInitException, ORTModuleTorchModelException},
        }
        self.policy = policy
        self.retry = retry
        self._exception = None
        self._raised_fallback_exception = False
        self._logger = logger

    def handle_exception(
        self, exception: Exception, log_level: _logger.LogLevel, override_policy: Optional[_FallbackPolicy] = None
    ) -> None:
        """Process incoming `exception` based on the selected `policy`

        If the incoming `exception` is handled by the specified policy, `_FallbackManager`
        saves the exception so that ORTModule can track the pending fallback
        and trigger it during model execution.
        Otherwise, the incoming exception is immediately raised.

        Args:
            exception (`ORTModuleFallbackException`): Exception that must be handled
            override_policy (`_FallbackPolicy`, optional): Policy to be checked for the incoming `exception`.
                if None is specified, all (except _FallbackPolicy.FALLBACK_DISABLE) are implicitly checked

        Raises:
            `exception`: Original exception is raised when there is no matching policy for it
        """

        def _set_exception(policy: _FallbackPolicy, exception: Exception, log_level: _logger.LogLevel):
            if (
                policy is not _FallbackPolicy.FALLBACK_DISABLE
                and self.policy.is_set(policy)
                and (
                    policy.value in self._policy_exception_map
                    and type(exception) in self._policy_exception_map[policy.value]
                )
            ):
                self._logger.info(f"Fallback for policy {policy.name} is pending.")

                # ORTModuleInitException exceptions do not call `fallback()` through `GraphExecutionManager`,
                # Instead, it fallbacks to PyTorch implicitly through `ORTModule._torch_module = TorchModulePytorch(module)`
                if policy == _FallbackPolicy.FALLBACK_BAD_INITIALIZATION:
                    self._logger.warning(
                        f"Fallback to PyTorch due to exception {type(exception)} was triggered. "
                        "Report this issue with a minimal repro at https://www.github.com/microsoft/onnxruntime. "
                        f"See details below:\n\n{_utils.get_exception_as_string(exception)}"
                    )

                self._exception = exception

        if override_policy is None:
            for policy in _FallbackPolicy:
                _set_exception(policy, exception, log_level)
        else:
            _set_exception(override_policy, exception, log_level)

        if self._exception is None:
            # No fallback, raise failure to user
            raise exception

    def is_pending(self) -> bool:
        """Returns True when a fallback is pending

        ORTModule must execute fallback to PyTorch engine when a pending fallback is detected
        """

        return self._exception is not None

    def fallback(self, log_level: _logger.LogLevel, *inputs, **kwargs):
        """Executes user PyTorch `model` using the provided inputs and return the result"""

        assert self.is_pending(), "`fallback` can only be called when there is a pending fallback"

        if not self._raised_fallback_exception:
            exception_type = type(self._exception)
            exception_string = _utils.get_exception_as_string(self._exception)

            # This warning will not be raised again if retry is not enabled
            self._logger.warning(
                f"Fallback to PyTorch due to exception {exception_type} was triggered. "
                "Report this issue with a minimal repro at https://www.github.com/microsoft/onnxruntime. "
                f"See details below:\n\n{exception_string}"
            )

            self._raised_fallback_exception = True

        # Pending fallbacks are reset to enforce retries
        if self.retry:
            self._raised_fallback_exception = False
            self._exception = None
        return self._original_module(*inputs, **kwargs)
