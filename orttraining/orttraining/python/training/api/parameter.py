# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# parameter.py

from onnxruntime.capi import _pybind_state as C


class Parameter:
    """
    A data structure that stores the parameters.
    """

    def __init__(self, parameter) -> None:
        if isinstance(parameter, C.OrtValue):
            self._parameter = parameter
        else:
            # An end user won't hit this error
            raise ValueError(
                "`Provided parameter` needs to be of type " + "`onnxruntime.capi.onnxruntime_pybind11_state.Parameter`"
            )

    def get_data(self) -> C.OrtValue:
        """
        Get the data from the parameter.
        """
        return self._parameter.get_data()

    def get_gradient(self) -> C.OrtValue:
        """
        Get the gradient from the parameter.
        """
        return self._parameter.get_gradient()

    def get_name(self) -> str:
        """
        Get the name of the parameter.
        """
        return self._parameter.get_name()
