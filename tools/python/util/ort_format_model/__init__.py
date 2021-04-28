# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys

# need to add the path to the ORT flatbuffers python module before we import anything else here
script_path = os.path.dirname(os.path.realpath(__file__))
ort_root = os.path.join(script_path, '..', '..', '..', '..')
ort_fbs_py_path = os.path.abspath(os.path.join(ort_root, 'onnxruntime', 'core', 'flatbuffers'))
sys.path.append(ort_fbs_py_path)

from .utils import create_config_from_models  # noqa
from .ort_model_processor import OrtFormatModelProcessor  # noqa
from .operator_type_usage_processors import (  # noqa
    GloballyAllowedTypesOpTypeImplFilter,
    OpTypeImplFilterInterface,
    OperatorTypeUsageManager)
