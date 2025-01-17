# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _fallback_exceptions.py


class ORTModuleFallbackException(Exception):  # noqa: N818
    """Base exception class for fallback

    Although it must be specialized for specific scenarios,
    it can also be used for generic exception that require fallback
    """


class ORTModuleInitException(ORTModuleFallbackException):
    """Trigger fallback for ORTModule initialization related exceptions

    This exception is triggered when an incompatible or missing requirements for ORTModule are detected,
    including PyTorch version, missing ORTModule's PyTorch C++ extension binaries, etc.
    """


class ORTModuleDeviceException(ORTModuleFallbackException):
    """Trigger fallback for device related exceptions

    NOTE: This exception is raised during device validation within ORTModule frontend.
    Some device related exceptions can only be detected during PyTorch ONNX exporter execution.
    This exception does not capture these scenarios.
    """


class ORTModuleIOError(ORTModuleFallbackException):
    """Trigger fallback for I/O related exceptions

    NOTE: This exception is raised during I/O validation within ORTModule Frontend.
    Some I/O related exceptions can only be detected during PyTorch ONNX exporter execution.
    This exception does not capture these scenarios.
    """


class ORTModuleTorchModelException(ORTModuleFallbackException):
    """Trigger fallback for PyTorch modules related exceptions

    This exception is raised during model validation within ORTModule frontend and is based on
    checking type(model) over a hardcoded list of incompatible models.
    """


class ORTModuleONNXModelException(ORTModuleFallbackException):
    """Trigger fallback for ONNX model related exceptions

    This exception is raised during model conversion to ONNX and post-processing validation within ORTModule frontend.
    """


def wrap_exception(
    new_exception: ORTModuleFallbackException, raised_exception: Exception
) -> ORTModuleFallbackException:
    """Wraps `raised_exception` exception as cause for the returned `new_exception` exception"""

    exception = None
    try:
        raise new_exception(raised_exception) from raised_exception
    except Exception as e:
        exception = e
    return exception
