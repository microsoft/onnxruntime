# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _logger

import os
import torch
import traceback
import warnings

from enum import IntFlag
from typing import Optional


class _FallbackPolicy(IntFlag):
    '''Policy to trigger fallback from ONNX Runtime engine to PyTorch

    Each policy can be combined with the others (using |) in order to aggregate them
    '''

    FALLBACK_DISABLE = 1
    FALLBACK_FORCE_TORCH_FORWARD = 2
    FALLBACK_FORCE_TORCH_BACKWARD = 4
    FALLBACK_UNSUPPORTED_DEVICE = 8
    FALLBACK_UNSUPPORTED_DATA = 16
    FALLBACK_UNSUPPORTED_TORCH_MODEL = 32
    FALLBACK_UNSUPPORTED_ONNX_MODEL = 64
    FALLBACK_BAD_INITIALIZATION = 128

    def is_set(self, policy):
        '''Check whether `policy` is set on the `_FallbackPolicy` instance

        FALLBACK_DISABLE implies the check will always fail and return False
        '''

        return not self.is_disabled() and policy in self

    def is_disabled(self):
        '''Check whether `_FallbackPolicy.FALLBACK_DEVICE` is set on the `_FallbackPolicy` instance'''

        return _FallbackPolicy.FALLBACK_DISABLE in self


class ORTModuleFallbackException(Exception):
    '''Base exception class for fallback

    Although it must be specialized for specific scenarios,
    it can also be used for generic exception that require fallback
    '''

    pass


class ORTModuleInitException(ORTModuleFallbackException):
    '''Trigger fallback for ORTModule initialization related exceptions

    This exception is triggered when an incompatible or missing requirements for ORTModule are detected,
    including PyTorch version, missing ORTModule's PyTorch C++ extension binaries, etc.
    '''

    pass


class ORTModuleDeviceException(ORTModuleFallbackException):
    '''Trigger fallback for device related exceptions

    NOTE: This exception is raised during device validation within ORTModule frontend.
    Some device related exceptions can only be detected during PyTorch ONNX exporter execution.
    This exception does not capture these scenarios.
    '''

    pass


class ORTModuleIOError(ORTModuleFallbackException):
    '''Trigger fallback for I/O related exceptions

    NOTE: This exception is raised during I/O validation within ORTModule Frontend.
    Some I/O related exceptions can only be detected during PyTorch ONNX exporter execution.
    This exception does not capture these scenarios.
    '''

    pass


class ORTModuleTorchModelException(ORTModuleFallbackException):
    '''Trigger fallback for PyTorch modules related exceptions

    This exception is raised during model validation within ORTModule frontend and is based on
    checking type(model) over a hardcoded list of incompatible models.
    '''

    pass


class ORTModuleONNXModelException(ORTModuleFallbackException):
    '''Trigger fallback for ONNX model related exceptions

    This exception is raised during model conversion to ONNX and post-processing validation within ORTModule frontend.
    '''

    pass


class _FallbackManager(object):
    '''Manages fallbacks based on incoming exceptions and specified policies

    The basic algorithm is based on a dictionary whose keys are the supported fallback policies
    and and values are a set of Exception that must be detected.

    When an exception that matches one of the enabled policies are detected,
    a fallback will be pending to execute by ORTModule frontend. If `retry` is False,
    ORTModule will fallback to PyTorch on the following steps. Otherwise, ORTModule [re]try
    to run the model using ORT backend in every step


    On the other hand, when the exception doesn't match any enabled policy, the exception will
    be raised to the user, terminating execution
    '''

    def __init__(self,
                 policy: _FallbackPolicy,
                 retry: bool):

        # Read policy from environment variable for testing purposes

        policy = os.getenv('ORTMODULE_FALLBACK_POLICY', policy)
        if isinstance(policy, str):
            policy = _FallbackPolicy[policy]

        # Read retry from environment variable for testing purposes
        retry = os.getenv('ORTMODULE_FALLBACK_RETRY', str(retry)).lower() in ['true', '1', 'yes']

        self._policy_exception_map = {_FallbackPolicy.FALLBACK_FORCE_TORCH_FORWARD.value: {ORTModuleFallbackException,
                                                                                           ORTModuleDeviceException,
                                                                                           ORTModuleIOError,
                                                                                           ORTModuleTorchModelException,
                                                                                           ORTModuleONNXModelException},
                                      _FallbackPolicy.FALLBACK_FORCE_TORCH_BACKWARD.value: {ORTModuleFallbackException,
                                                                                            ORTModuleDeviceException,
                                                                                            ORTModuleIOError,
                                                                                            ORTModuleTorchModelException,
                                                                                            ORTModuleONNXModelException},
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE.value: {ORTModuleDeviceException},
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_DATA.value: {ORTModuleIOError},
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_TORCH_MODEL.value: {ORTModuleTorchModelException},
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_ONNX_MODEL.value: {ORTModuleONNXModelException},
                                      _FallbackPolicy.FALLBACK_BAD_INITIALIZATION.value: {ORTModuleInitException,
                                                                                          ORTModuleTorchModelException},
                                      }
        self.policy = policy
        self.retry = retry
        self._exception = None

    def handle_exception(self,
                         exception: Exception,
                         log_level: _logger.LogLevel,
                         override_policy: Optional[_FallbackPolicy] = None) -> None:
        '''Process incoming `exception` based on the selected `policy`

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
        '''
        def _set_exception(policy: _FallbackPolicy, exception: Exception, log_level: _logger.LogLevel):
            if policy is not _FallbackPolicy.FALLBACK_DISABLE and \
                    self.policy.is_set(policy) and \
                    (policy.value in self._policy_exception_map and type(exception) in self._policy_exception_map[policy.value]):

                if log_level <= _logger.LogLevel.INFO:
                    warnings.warn(
                        f'Fallback for policy {policy.name} is pending.', UserWarning)
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
        '''Returns True when a fallback is pending

        ORTModule must execute fallback to PyTorch engine when a pending fallback is detected
        '''

        return self._exception is not None

    def fallback(self, model: torch.nn.Module, log_level: _logger.LogLevel, *inputs, **kwargs):
        '''Executes user PyTorch `model` using the provided inputs and return the result'''

        assert self.is_pending(), '`fallback` can only be called when there is a pending fallback'

        if log_level <= _logger.LogLevel.WARNING:
            warnings.warn(
                (f'Fallback to PyTorch due to exception {type(self._exception)} was triggered. '
                 'Report this issue with a minimal repro at https://www.github.com/microsoft/onnxruntime. '
                 f'See details below:\n\n{get_exception_as_string(self._exception)}'), UserWarning)

        # Pending fallbacks are resetted to enforce retries
        if self.retry:
            self._exception = None
        return model(*inputs, **kwargs)


def wrap_exception(new_exception: ORTModuleFallbackException, raised_exception: Exception) -> ORTModuleFallbackException:
    '''Wraps `raised_exception` exception as cause for the returned `new_exception` exception'''

    exception = None
    try:
        raise new_exception(raised_exception) from raised_exception
    except Exception as e:
        exception = e
    return exception

def get_exception_as_string(exception):
    assert isinstance(exception, Exception), 'exception must be a `Exception`'

    try:
        raise exception
    except:
        return traceback.format_exc()
