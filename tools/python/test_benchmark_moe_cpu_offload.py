import unittest
from types import SimpleNamespace

from benchmark_moe_cpu_offload import decode_summary, model_session_options


class TestMoeOffloadBenchmark(unittest.TestCase):
    def options(self, **overrides):
        values = {
            "cpu_offload_experts": 1000,
            "counter_alpha": 0.9,
            "swap_epsilon": 0.1,
            "swap": True,
            "counter_log": False,
            "state": None,
        }
        values.update(overrides)
        return model_session_options(SimpleNamespace(**values))

    def test_adaptive_swaps_remain_enabled_without_logging(self):
        self.assertEqual(
            self.options(),
            {
                "log_severity_level": 3,
                "session.enable_moe_expert_statistics": "0",
                "session.moe_cpu_offload_experts": "1000",
                "session.moe_expert_counter_alpha": "0.9",
                "session.moe_expert_swap_epsilon": "0.1",
                "session.moe_max_in_flight_swaps_per_device": "4",
            },
        )

    def test_zero_cpu_experts_is_an_uninstrumented_all_gpu_reference(self):
        self.assertEqual(
            self.options(cpu_offload_experts=0),
            {"log_severity_level": 3, "session.enable_moe_expert_statistics": "0"},
        )

    def test_disabled_swaps_preserve_cpu_offload(self):
        options = self.options(swap=False)
        self.assertEqual(options["session.moe_cpu_offload_experts"], "1000")
        self.assertEqual(options["session.moe_max_in_flight_swaps_per_device"], "0")

    def test_zero_epsilon_is_preserved(self):
        self.assertEqual(self.options(swap_epsilon=0)["session.moe_expert_swap_epsilon"], "0")

    def test_decode_excludes_generation_call_using_prefill_logits(self):
        summary = decode_summary([100, 0.01, 0.03])
        self.assertAlmostEqual(summary["mean_ms"], 20)
        self.assertAlmostEqual(summary["median_ms"], 20)
        self.assertAlmostEqual(summary["min_ms"], 10)
        self.assertAlmostEqual(summary["max_ms"], 30)

    def test_decode_requires_actual_inference(self):
        with self.assertRaisesRegex(ValueError, "empty timing sample"):
            decode_summary([100])


if __name__ == "__main__":
    unittest.main()
