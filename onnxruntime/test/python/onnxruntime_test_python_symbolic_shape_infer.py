# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import sys
from pathlib import Path

class TestSymbolicShapeInference(unittest.TestCase):
    def test_symbolic_shape_infer(self):
        cwd = os.getcwd()
        test_model_dir = os.path.join(cwd, '..', 'models')
        for filename in Path(test_model_dir).rglob('*.onnx'):
            if filename.name.startswith('.'):
                continue  # skip some bad model files
            print("Running symbolic shape inference on : " + str(filename))
            SymbolicShapeInference.infer_shapes(
                input_model=str(filename),
                output_model=None,
                auto_merge=True,
                int_max=100000,
                guess_output_rank=True)

if __name__ == '__main__':
    unittest.main()
