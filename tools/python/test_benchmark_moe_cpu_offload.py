import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark_moe_cpu_offload import decode_summary, effective_counter_parameters, model_session_options, parse_args


class TestMoeOffloadBenchmark(unittest.TestCase):
    def options(self, **overrides):
        values = {
            "cpu_offload_experts": 1000,
            "counter_alpha": 0.9,
            "swap_epsilon": 0.1,
            "swap": True,
            "counter_log": False,
            "state": None,
            "popularity_output": None,
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

    def test_popularity_output_is_forwarded_to_session(self):
        self.assertEqual(
            self.options(popularity_output="popularities.json")["session.moe_expert_popularity_file"],
            "popularities.json",
        )

    def test_state_parameters_override_cli_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_path = Path(temporary_directory) / "state.json"
            state_path.write_text(json.dumps({"alpha": 0.4, "epsilon": 0.2, "experts": []}), encoding="utf-8")
            args = SimpleNamespace(state=state_path, counter_alpha=0.9, swap_epsilon=0.1)
            self.assertEqual(effective_counter_parameters(args), (0.4, 0.2))
            options = self.options(state=state_path)
            self.assertNotIn("session.moe_expert_counter_alpha", options)
            self.assertNotIn("session.moe_expert_swap_epsilon", options)

    def test_decode_excludes_generation_call_using_prefill_logits(self):
        summary = decode_summary([100, 0.01, 0.03])
        self.assertAlmostEqual(summary["mean_ms"], 20)
        self.assertAlmostEqual(summary["median_ms"], 20)
        self.assertAlmostEqual(summary["min_ms"], 10)
        self.assertAlmostEqual(summary["max_ms"], 30)

    def test_decode_requires_actual_inference(self):
        with self.assertRaisesRegex(ValueError, "empty timing sample"):
            decode_summary([100])

    def test_cli_rejects_invalid_measurement_options(self):
        arguments = [
            "--name",
            "test",
            "--model",
            "model",
            "--prompts",
            "prompts.json",
            "--output",
            "result.json",
            "--log",
            "run.log",
            "--cpu-offload-experts",
            "1000",
            "--swap",
            "--no-counter-log",
            "--no-cuda-graph",
        ]
        for option, value in [
            ("--max-new-tokens", "1"),
            ("--cpu-offload-experts", "-1"),
            ("--counter-alpha", "nan"),
            ("--counter-alpha", "1.1"),
            ("--swap-epsilon", "inf"),
            ("--swap-epsilon", "-0.1"),
            ("--log", "result.json"),
        ]:
            with self.subTest(option=option, value=value):
                error_output = io.StringIO()
                with contextlib.redirect_stderr(error_output), self.assertRaises(SystemExit) as error:
                    parse_args([*arguments, option, value])
                self.assertEqual(error.exception.code, 2)
                self.assertIn(option, error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
