# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .get_azcopy import get_azcopy
from .logger import get_logger
from .platform import (is_windows, is_macOS, is_linux)
from .run import run

try:
    import flatbuffers  # noqa
    from .reduced_build_config_parser import parse_config
except ImportError:
    get_logger('tools_python_utils').info('flatbuffers module is not installed. parse_config will not be available')
