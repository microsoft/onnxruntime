# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import torch
from onnxruntime_test_ort_trainer import runBertTrainingTest

class TestOrtTrainer(unittest.TestCase):
    def testBertTrainingMixedPrecision(self):
        torch.manual_seed(1)
        expected_losses = [11.078125, 11.0, 11.0390625, 11.0, 11.015625, 11.0, 10.9921875, 11.0703125]
        expected_all_finites = [False, True, True, True, True, True, True, True]
        expected_eval_loss = [11.046875]
        actual_losses, actual_all_finites, actual_eval_loss = runBertTrainingTest(
            gradient_accumulation_steps=1, use_mixed_precision=True, allreduce_post_accumulation=False, use_simple_model_desc=False)

        rtol = 1e-01
        assert_allclose(expected_losses, actual_losses, rtol=rtol, err_msg="loss mismatch")
        assert_array_equal(expected_all_finites, actual_all_finites, "all_finite mismatch")
        assert_allclose(expected_eval_loss, actual_eval_loss, rtol=rtol, err_msg="evaluation loss mismatch")

    def testBertTrainingMixedPrecisionInternalLossScale(self):
        torch.manual_seed(1)
        expected_losses = [11.078125, 11.0, 11.0390625, 11.0, 11.015625, 11.0, 10.9921875, 11.0703125]
        expected_all_finites = [False, True, True, True, True, True, True, True]
        expected_eval_loss = [11.046875]
        actual_losses, actual_all_finites, actual_eval_loss = runBertTrainingTest(
            gradient_accumulation_steps=1,
            use_mixed_precision=True,
            allreduce_post_accumulation=False,
            use_simple_model_desc=False,
            use_internel_loss_scale=True)

        rtol = 1e-01
        assert_allclose(expected_losses, actual_losses, rtol=rtol, err_msg="loss mismatch")
        assert_array_equal(expected_all_finites, actual_all_finites, "all_finite mismatch")
        assert_allclose(expected_eval_loss, actual_eval_loss, rtol=rtol, err_msg="evaluation loss mismatch")

    def testBertTrainingGradientAccumulationMixedPrecision(self):
        torch.manual_seed(1)
        expected_losses = [11.046875, 11.171875, 11.0234375, 11.046875, 10.8984375, 10.9921875, 11.078125, 10.96875]
        expected_all_finites = [False, True]
        expected_eval_loss = [11.0546875]
        actual_losses, actual_all_finites, actual_eval_loss = runBertTrainingTest(
            gradient_accumulation_steps=4, use_mixed_precision=True, allreduce_post_accumulation=False, use_simple_model_desc=False)

        rtol = 1e-01
        assert_allclose(expected_losses, actual_losses, rtol=rtol, err_msg="loss mismatch")
        assert_array_equal(expected_all_finites, actual_all_finites, "all_finite mismatch")
        assert_allclose(expected_eval_loss, actual_eval_loss, rtol=rtol, err_msg="evaluation loss mismatch")

if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)
