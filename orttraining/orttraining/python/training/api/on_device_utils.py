# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# checkpoint_utils.py

import numpy as np

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi._pybind_state import get_parameters_difference as _internal_get_parameters_difference


def get_parameters_difference(output_params, old_output_params):
    """Calculates the difference between parameters and returns it."""
    output = C.OrtValue.ortvalue_from_numpy(
        np.array([]),
        C.OrtDevice(
            C.OrtDevice.cpu(),
            C.OrtDevice.default_memory(),
            0,
        ),
    )

    _internal_get_parameters_difference(output_params, old_output_params, output)

    return output
