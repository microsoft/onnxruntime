import unittest

from benchmark_gpu_decode_stack import run_request
from benchmark_gpu_slice import run_request as verified_request
from run_gpu_decode_stack_model import selectors
from run_gpu_decode_stack_pipeline import compare


def sample(decode, ttft=4):
    return dict(decode_tps=decode, decode_seconds=255 / decode, output_tokens=256, ttft_seconds=ttft)


class DecodeStackTests(unittest.TestCase):
    def test_fc2_comparison_keeps_greedy_and_slice_fixed(self):
        self.assertEqual(selectors("fc2-private-a-stack-v1", 2048, 0), (1, 1, 0))
        self.assertEqual(selectors("fc2-private-a-stack-v1", 2048, 1), (1, 1, 1))

    def test_32k_combined_uses_supported_cpu_control(self):
        self.assertEqual(selectors("gpu-slice-combined32k-v1", 32768, 0), (0, 0, 0))
        self.assertEqual(selectors("gpu-slice-combined32k-v1", 32768, 1), (1, 1, 0))
        with self.assertRaises(ValueError):
            selectors("gpu-slice-combined32k-v1", 2048, 0)

    def test_verified_request_is_reused_without_edits(self):
        self.assertIs(run_request, verified_request)

    def test_decode_gain_cannot_override_ttft_regression(self):
        control = [sample(82)] * 3
        self.assertTrue(compare([sample(83, 4.08)] * 3, control, control)["supported"])
        self.assertFalse(compare([sample(83, 4.081)] * 3, control, control)["supported"])

    def test_each_decode_request_must_improve(self):
        control = [sample(82)] * 3
        self.assertFalse(compare([sample(90), sample(90), sample(81)], control, control)["supported"])
        self.assertFalse(compare(control, control, control)["supported"])

    def test_invalid_count_rejects(self):
        control = [sample(82)] * 3
        invalid = sample(83)
        invalid["output_tokens"] = 255
        with self.assertRaises(RuntimeError):
            compare([invalid] * 3, control, control)


if __name__ == "__main__":
    unittest.main()
