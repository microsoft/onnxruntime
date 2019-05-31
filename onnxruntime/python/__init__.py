#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from onnxruntime.capi import _pybind_state as C

def set_logging_severity(severity):
    """
    :param severity" Set the default logging severity.
                     This must be done prior to creating the first InferenceSession, and once set is fixed.
                     0 = Verbose, 1 = Info, 2 = Warning, 3 = Error, 4 = Fatal.
    """
    
    C.set_logging_severity(severity)

