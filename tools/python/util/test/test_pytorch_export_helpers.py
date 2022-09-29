# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import torch

from ..pytorch_export_helpers import infer_input_info

# example usage from <ort root>/tools/python
# python -m unittest util/test/test_pytorch_export_helpers.py
# NOTE: at least on Windows you must use that as the working directory for all the imports to be happy


class TestModel(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TestModel, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x, min=0, max=1):
        step1 = self.linear1(x).clamp(min=min, max=max)
        step2 = self.linear2(step1)
        return step2


class TestInferInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._model = TestModel(1000, 100, 10)
        cls._input = torch.randn(1, 1000)

    def test_positional(self):
        # test we can infer the input names from the forward method when positional args are used
        input_names, inputs_as_tuple = infer_input_info(self._model, self._input, 0, 1)
        self.assertEqual(input_names, ["x", "min", "max"])

    def test_keywords(self):
        # test that we sort keyword args and the inputs to match the module
        input_names, inputs_as_tuple = infer_input_info(self._model, self._input, max=1, min=0)
        self.assertEqual(input_names, ["x", "min", "max"])
        self.assertEqual(inputs_as_tuple, (self._input, 0, 1))
