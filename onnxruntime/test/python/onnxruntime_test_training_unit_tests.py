# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import torch
import torch.nn as nn
from numpy.testing import assert_allclose
from onnxruntime_test_ort_trainer import map_optimizer_attributes, ort_trainer_learning_rate_description
from onnxruntime_test_training_unittest_utils import process_dropout

import onnxruntime
from onnxruntime.capi.ort_trainer import IODescription, ModelDescription, ORTTrainer


class TestTrainingDropout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        onnxruntime.set_seed(1)

    @unittest.skip(
        "Temporarily disable this test. The graph below will trigger ORT to "
        "sort backward graph before forward graph which gives incorrect result. "
        "https://github.com/microsoft/onnxruntime/issues/16801"
    )
    def test_training_and_eval_dropout(self):
        class TwoDropoutNet(nn.Module):
            def __init__(self, drop_prb_1, drop_prb_2, dim_size):
                super().__init__()
                self.drop_1 = nn.Dropout(drop_prb_1)
                self.drop_2 = nn.Dropout(drop_prb_2)
                self.weight_1 = torch.nn.Parameter(torch.zeros(dim_size, dtype=torch.float32))

            def forward(self, x):
                x = x + self.weight_1
                x = self.drop_1(x)
                x = self.drop_2(x)
                output = x
                return output[0]

        dim_size = 3
        device = torch.device("cuda", 0)
        # This will drop all values, therefore expecting all 0 in output tensor
        model = TwoDropoutNet(0.999, 0.999, dim_size)
        input_desc = IODescription("input", [dim_size], torch.float32)
        output_desc = IODescription("output", [], torch.float32)
        model_desc = ModelDescription([input_desc], [output_desc])
        lr_desc = ort_trainer_learning_rate_description()
        model = ORTTrainer(
            model,
            None,
            model_desc,
            "LambOptimizer",
            map_optimizer_attributes,
            lr_desc,
            device,
            postprocess_model=process_dropout,
            world_rank=0,
            world_size=1,
        )
        input = torch.ones(dim_size, dtype=torch.float32).to(device)
        expected_training_output = [0.0]
        expected_eval_output = [1.0]
        learning_rate = torch.tensor([1.0000000e00]).to(device)
        input_args = [input, learning_rate]
        train_output = model.train_step(*input_args)

        rtol = 1e-04
        assert_allclose(
            expected_training_output,
            train_output.item(),
            rtol=rtol,
            err_msg="dropout training loss mismatch",
        )

        eval_output = model.eval_step(input)
        assert_allclose(
            expected_eval_output,
            eval_output.item(),
            rtol=rtol,
            err_msg="dropout eval loss mismatch",
        )

        # Do another train step to make sure it's using original ratios
        train_output_2 = model.train_step(*input_args)
        assert_allclose(
            expected_training_output,
            train_output_2.item(),
            rtol=rtol,
            err_msg="dropout training loss 2 mismatch",
        )


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
