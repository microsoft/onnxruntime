# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .get_azcopy import get_azcopy  # noqa: F401
from .logger import get_logger
from .platform_helpers import is_linux, is_macOS, is_windows  # noqa: F401
from .run import run  # noqa: F401

try:
    import flatbuffers  # noqa: F401

    from .reduced_build_config_parser import parse_config  # noqa: F401
except ImportError:
    get_logger("tools_python_utils").info("flatbuffers module is not installed. parse_config will not be available")

# see if we can make the pytorch helpers available.
import importlib.util

have_torch = importlib.util.find_spec("torch")
if have_torch:
    from .pytorch_export_helpers import infer_input_info  # noqa: F401
