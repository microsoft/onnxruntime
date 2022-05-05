# SPDX-License-Identifier: Apache-2.0

import sys

import onnx.defs
import numpy as np  # type: ignore
from onnx import ModelProto
from typing import List, Optional, Text, Sequence
from onnx.backend.test.case.utils import import_recursive
from onnx.backend.test.case.test_case import TestCase

_SimpleModelTestCases = []


def expect(model: ModelProto,
           inputs: Sequence[np.ndarray],
           outputs: Sequence[np.ndarray],
           name: Optional[Text] = None,
           ) -> None:
    name = name or model.graph.name
    _SimpleModelTestCases.append(
        TestCase(
            name=name,
            model_name=model.graph.name,
            url=None,
            model_dir=None,
            model=model,
            data_sets=[(inputs, outputs)],
            kind='simple',
            rtol=1e-3,
            atol=1e-7,
        ))


def collect_testcases() -> List[TestCase]:
    '''Collect model test cases defined in python/numpy code and in model zoo.
    '''    
    import_recursive(sys.modules[__name__])
    return _SimpleModelTestCases
