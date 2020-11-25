# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import platform
import warnings
import onnxruntime.capi._ld_preload  # noqa: F401

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import *  # noqa
except ImportError as e:
    warnings.warn("Cannot load onnxruntime.capi. Error: '{0}'.".format(str(e)))

    # If on Windows, check if this import error is caused by the user not installing the 2019 VC Runtime
    # The VC Redist installer usually puts the VC Runtime dlls in the System32 folder
    # This may not always paint the true picture as anyone building from source using VS 2017 might hit this error
    # because the machine might be missing the 2019 VC Runtime but it is not actually needed in that case and the
    # import error might actually be due to some other reason.
    # TODO: Add a guard against False Positive error message
    # As a proxy for checking if the 2019 VC Runtime is installed,
    # we look for a specific dll only shipped with the 2019 VC Runtime
    if platform.system().lower() == 'windows' and not os.path.isfile('c:\\Windows\\System32\\vcruntime140_1.dll'):
        warnings.warn("Unless you have built the wheel using VS 2017, "
                      "please install the 2019 Visual C++ runtime and then try again")
