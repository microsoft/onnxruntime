# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "models", "gpt2"))

import convert_to_onnx  # noqa: E402, F401

# added for backward compatible
import gpt2_helper  # noqa: E402, F401

sys.path.append(os.path.join(os.path.dirname(__file__), "models", "t5"))
