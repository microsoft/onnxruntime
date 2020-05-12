#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
import sys
import os
import platform
import warnings
import onnxruntime.capi._ld_preload

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import *  # noqa
except ImportError as e:
    warnings.warn("Cannot load onnxruntime.capi. Error: '{0}'.".format(str(e)))
    
	# If on Windows, check if this import error is caused by the user not installing the 2019 VC Runtime
    if platform.system().lower() == 'windows':
        # The VC Redist installer usually puts the VC Runtime dlls in the System32 folder
        if not os.path.isfile('c:\\Windows\\System32\\vcruntime140_1.dll'):
            warnings.warn("Please install the Visual C++ 2019 runtime and then try again")
            