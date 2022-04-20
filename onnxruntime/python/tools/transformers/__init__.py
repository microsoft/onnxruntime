#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'gpt2'))

# added for backward compatible
import gpt2_helper
import convert_to_onnx
