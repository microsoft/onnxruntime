import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark_moe_cpu_offload import (
    MemorySampler,
    decode_summary,
    effective_counter_parameters,
    model_session_options,
    parse_args,
    physical_gpu_index,
    summarize_measurements,
)


class TestMoeOffloadBenchmark(unittest.TestCase):
    def test_cpu_memory_sampling_without_cuda(self):
        for visible_devices in ("", "-1", "GPU-unavailable", "0,1"):
            with self.subTest(visible_devices=visible_devices):
                gpu_index = physical_gpu_index("cpu", visible_devices)
                sampler = MemorySampler(gpu_index)
                self.addCleanup(lambda sampler=sampler: sampler.stop() if sampler.thread.is_alive() else None)
                self.assertIsNone(sampler.nvml)
                sampler.set_phase("prompt_3")
                snapshot = sampler.snapshot()
                memory = sampler.stop()
                self.assertGreater(snapshot["cpu_rss_mib"], 0)
                self.assertIsNone(snapshot["gpu_used_mib"])
                self.assertGreaterEqual(memory["peak_cpu_rss_mib"], memory["baseline_cpu_rss_mib"])
                self.assertIsNone(memory["baseline_gpu_used_mib"])
                self.assertIsNone(memory["peak_gpu_used_mib"])
                self.assertIsNone(memory["phase_peaks"]["prompt_3"]["gpu_used_mib"])
                result = summarize_measurements(
                    [
                        {
                            "prompt_index": 3,
                            "generated_tokens": 3,
                            "prefill_ms": 1,
                            "steady_decode": {"mean_ms": 10},
                            "memory": memory["phase_peaks"]["prompt_3"],
                        }
                    ]
                )
                self.assertGreater(result["peak_cpu_rss_mib"], 0)
                self.assertIsNone(result["peak_gpu_used_mib"])
                self.assertAlmostEqual(result["decoded_tokens_per_second"], 100)

    def test_cuda_memory_sampling_requires_one_physical_gpu(self):
        self.assertEqual(physical_gpu_index("cuda", "2"), 2)
        for visible_devices in ("", "-1", "GPU-unavailable", "0,1"):
            with (
                self.subTest(visible_devices=visible_devices),
                self.assertRaisesRegex(ValueError, "one physical GPU index"),
            ):
                physical_gpu_index("cuda", visible_devices)

    def options(self, **overrides):
        values = {
            "cpu_offload_experts": 1000,
            "strategy": "adaptive",
            "gpu_ratio": 0.8,
            "layer_order": "first",
            "counter_alpha": 0.9,
            "swap_epsilon": 0.1,
            "max_in_flight_swaps_per_device": 4,
            "on_demand_threshold": 0,
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
                "session.moe_cpu_offload_strategy": "adaptive",
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

    def test_basic_strategy_is_forwarded_with_nonzero_swap_limit(self):
        options = self.options(strategy="basic")
        self.assertEqual(options["session.moe_cpu_offload_strategy"], "basic")
        self.assertEqual(options["session.moe_max_in_flight_swaps_per_device"], "4")

    def test_layered_strategy_forwards_ratio_and_swaps(self):
        for ratio in (0.8, 0.5, 1):
            with self.subTest(ratio=ratio):
                options = self.options(strategy="layered", gpu_ratio=ratio)
                self.assertEqual(options["session.moe_cpu_offload_strategy"], "layered")
                self.assertEqual(options["session.moe_cpu_offload_gpu_ratio"], str(ratio))
                self.assertEqual(options["session.moe_max_in_flight_swaps_per_device"], "4")
        self.assertNotIn("session.moe_cpu_offload_gpu_ratio", self.options(strategy="basic"))
        self.assertNotIn("session.moe_cpu_offload_gpu_ratio", self.options(strategy="layered", cpu_offload_experts=0))

    def test_swap_limit_is_forwarded_to_session(self):
        self.assertEqual(
            self.options(max_in_flight_swaps_per_device=16)["session.moe_max_in_flight_swaps_per_device"],
            "16",
        )

    def test_on_demand_threshold_is_independent_of_background_swaps(self):
        options = self.options(on_demand_threshold=2, swap=False)
        self.assertEqual(options["session.moe_cpu_offload_on_demand_threshold"], "2")
        self.assertEqual(options["session.moe_max_in_flight_swaps_per_device"], "0")
        self.assertNotIn("session.moe_cpu_offload_on_demand_threshold", self.options())

    def test_layer_order_is_forwarded_only_for_basic_and_layered(self):
        for strategy in ("basic", "layered"):
            for order in ("first", "last", "middle"):
                with self.subTest(strategy=strategy, order=order):
                    options = self.options(strategy=strategy, layer_order=order)
                    self.assertEqual(options["session.moe_cpu_offload_layer_order"], order)
        self.assertNotIn("session.moe_cpu_offload_layer_order", self.options())
        self.assertNotIn(
            "session.moe_cpu_offload_layer_order",
            self.options(strategy="layered", layer_order="last", cpu_offload_experts=0),
        )

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

    def test_measurements_use_only_selected_prompt_timings_and_memory(self):
        prompts = [
            {
                "prompt_index": index,
                "generated_tokens": 256,
                "prefill_ms": 1000 if index <= 2 else index,
                "steady_decode": {"mean_ms": 1000 if index <= 2 else 10},
                "memory": {
                    "gpu_used_mib": 10000 if index <= 2 else index * 100,
                    "cpu_rss_mib": 20000 if index <= 2 else index * 10,
                },
            }
            for index in range(1, 6)
        ]
        result = summarize_measurements(prompts[2:5])
        self.assertEqual(result["prompt_indices"], [3, 4, 5])
        self.assertEqual(result["prefill_ms"], 4)
        self.assertEqual(result["decoded_tokens"], 762)
        self.assertAlmostEqual(result["decoded_tokens_per_second"], 100)
        self.assertEqual(result["peak_gpu_used_mib"], 500)
        self.assertEqual(result["peak_cpu_rss_mib"], 50)

    def test_measurements_reject_missing_samples(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            summarize_measurements([])
        with self.assertRaisesRegex(ValueError, "steady-state"):
            summarize_measurements([{"generated_tokens": 2}])

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
            ("--on-demand-threshold", "-1"),
            ("--max-new-tokens", "1"),
            ("--max-new-tokens", "2"),
            ("--cpu-offload-experts", "-1"),
            ("--counter-alpha", "nan"),
            ("--counter-alpha", "1.1"),
            ("--swap-epsilon", "inf"),
            ("--swap-epsilon", "-0.1"),
            ("--max-in-flight-swaps-per-device", "-1"),
            ("--strategy", "invalid"),
            ("--layer-order", "invalid"),
            ("--layer-order", "last"),
            ("--layer-order", "middle"),
            ("--gpu-ratio", "0"),
            ("--gpu-ratio", "-0.1"),
            ("--gpu-ratio", "1.1"),
            ("--gpu-ratio", "nan"),
            ("--gpu-ratio", "inf"),
            ("--log", "result.json"),
        ]:
            with self.subTest(option=option, value=value):
                error_output = io.StringIO()
                with contextlib.redirect_stderr(error_output), self.assertRaises(SystemExit) as error:
                    parse_args([*arguments, option, value])
                self.assertEqual(error.exception.code, 2)
                self.assertIn(option, error_output.getvalue())
        for extra in (["--strategy", "basic"], ["--cpu-offload-experts", "0"]):
            with self.subTest(extra=extra):
                error_output = io.StringIO()
                with contextlib.redirect_stderr(error_output), self.assertRaises(SystemExit) as error:
                    parse_args([*arguments, "--on-demand-threshold", "1", *extra])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--on-demand-threshold", error_output.getvalue())
        self.assertEqual(parse_args([*arguments, "--on-demand-threshold", "2"]).on_demand_threshold, 2)
        self.assertEqual(parse_args([*arguments, "--max-new-tokens", "3"]).max_new_tokens, 3)


if __name__ == "__main__":
    unittest.main()
