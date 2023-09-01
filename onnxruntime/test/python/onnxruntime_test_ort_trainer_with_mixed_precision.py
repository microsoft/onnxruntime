# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from numpy.testing import assert_allclose, assert_array_equal
from onnxruntime_test_ort_trainer import run_bert_training_test


class TestOrtTrainer(unittest.TestCase):
    def test_bert_training_mixed_precision(self):
        expected_losses = [
            11.034248352050781,
            11.125300407409668,
            11.006105422973633,
            11.047048568725586,
            11.027417182922363,
            11.015759468078613,
            11.060905456542969,
            10.971782684326172,
        ]
        expected_all_finites = [True, True, True, True, True, True, True, True]
        expected_eval_loss = [10.959012985229492]
        actual_losses, actual_all_finites, actual_eval_loss = run_bert_training_test(
            gradient_accumulation_steps=1,
            use_mixed_precision=True,
            allreduce_post_accumulation=False,
            use_simple_model_desc=False,
        )

        rtol = 1e-02
        assert_allclose(expected_losses, actual_losses, rtol=rtol, err_msg="loss mismatch")
        assert_array_equal(expected_all_finites, actual_all_finites, "all_finite mismatch")
        assert_allclose(
            expected_eval_loss,
            actual_eval_loss,
            rtol=rtol,
            err_msg="evaluation loss mismatch",
        )

    def test_bert_training_mixed_precision_internal_loss_scale(self):
        expected_losses = [
            11.034248352050781,
            11.125300407409668,
            11.006105422973633,
            11.047048568725586,
            11.027417182922363,
            11.015759468078613,
            11.060905456542969,
            10.971782684326172,
        ]
        expected_eval_loss = [10.959012985229492]
        actual_losses, actual_eval_loss = run_bert_training_test(
            gradient_accumulation_steps=1,
            use_mixed_precision=True,
            allreduce_post_accumulation=False,
            use_simple_model_desc=False,
            use_internel_loss_scale=True,
        )

        rtol = 1e-02
        assert_allclose(expected_losses, actual_losses, rtol=rtol, err_msg="loss mismatch")
        assert_allclose(
            expected_eval_loss,
            actual_eval_loss,
            rtol=rtol,
            err_msg="evaluation loss mismatch",
        )

    def test_bert_training_gradient_accumulation_mixed_precision(self):
        expected_losses = [
            11.034248352050781,
            11.125300407409668,
            11.006077766418457,
            11.047025680541992,
            11.027434349060059,
            11.0156831741333,
            11.060973167419434,
            10.971841812133789,
        ]
        expected_all_finites = [True, True]
        expected_eval_loss = [10.95903205871582]
        actual_losses, actual_all_finites, actual_eval_loss = run_bert_training_test(
            gradient_accumulation_steps=4,
            use_mixed_precision=True,
            allreduce_post_accumulation=False,
            use_simple_model_desc=False,
        )

        rtol = 1e-02
        assert_allclose(expected_losses, actual_losses, rtol=rtol, err_msg="loss mismatch")
        assert_array_equal(expected_all_finites, actual_all_finites, "all_finite mismatch")
        assert_allclose(
            expected_eval_loss,
            actual_eval_loss,
            rtol=rtol,
            err_msg="evaluation loss mismatch",
        )


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
