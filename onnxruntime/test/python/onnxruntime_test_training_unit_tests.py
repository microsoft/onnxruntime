# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest
import sys
import copy
from numpy.testing import assert_allclose, assert_array_equal

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from .onnxruntime_test_ort_trainer import map_optimizer_attributes, ort_trainer_learning_rate_description
from helper import get_name
import onnxruntime
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription, LossScaler, generate_sample

class TestTrainingDropout(unittest.TestCase):
    def TestTrainingAndEvalDropout(self):
        class TwoDropoutNet(nn.Module):
            def __init__(self, drop_prb_1, drop_prb_2):
                super(TwoDropoutNet, self).__init__()
                self.drop_1 = nn.Dropout(drop_prb_1)
                self.drop_2 = nn.Dropout(drop_prb_2)
            def forward(self, x):
                output = self.drop_1(x)
                output = self.drop_2(output)
                return output
        
        model = TwoDropoutNet(1, 1)
        input_desc = IODescription('input', ['dim_1', 'dim_2', 'dim_3'], torch.float32)
        output_desc = IODescription('output', ['dim_1', 'dim_2', 'dim_3'], torch.float32)
        model_desc = ModelDescription([input_desc], [output_desc])
        device = torch.device("cuda", 0)
        lr_desc = ort_trainer_learning_rate_description()
        model = ORTTrainer(model, None, model_desc, "LambOptimizer",
                        map_optimizer_attributes,
                        lr_desc,
                        device,
                        world_rank=0, world_size=1)
        input = torch.ones(3, dtype=torch.float32)
        expected_training_output = [1, 1, 1]
        expected_eval_output = [0, 0, 0]
        learning_rate = torch.tensor([1.0000000e+00]).to(device)
        train_output = model.train_step(input, learning_rate = learning_rate)

        rtol = 1e-03
        assert_allclose(expected_training_output, train_output, rtol=rtol, err_msg="dropout training loss mismatch")

        eval_output = model.eval_step(input)
        assert_allclose(expected_eval_output, eval_output, rtol=rtol, err_msg="dropout eval loss mismatch")




