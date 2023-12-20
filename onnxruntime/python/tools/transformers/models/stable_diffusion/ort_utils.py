# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import os
import sys

logger = logging.getLogger(__name__)


def add_transformers_dir_to_path():
    sys.path.append(os.path.dirname(__file__))

    transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if transformers_dir not in sys.path:
        sys.path.append(transformers_dir)


add_transformers_dir_to_path()

# Walkaround so that we can test local change without building new package
from io_binding_helper import CudaSession  # noqa
from onnx_model import OnnxModel  # noqa
