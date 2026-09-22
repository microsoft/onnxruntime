# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import ort_genai_workspace_benchmark as benchmark


class WorkspaceBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.config = {
            "model": {
                "context_length": 32768,
                "vocab_size": 100,
                "bos_token_id": 1,
                "pad_token_id": 0,
                "eos_token_id": [2, 3],
                "decoder": {
                    "session_options": {
                        "provider_options": [
                            {"cuda": {"enable_cuda_graph": "0", "enable_skip_layer_norm_strict_mode": "1"}}
                        ],
                        "ep.cuda.fpa_intb_gemm": "1",
                        "ep.cuda.gqa_workspace_max_total_sequence_length": "32768",
                    },
                    "inputs": {
                        "input_ids": "input_ids",
                        "attention_mask": "attention_mask",
                        "past_key_names": "past_key_values.%d.key",
                        "past_value_names": "past_key_values.%d.value",
                    },
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                    "head_size": 8,
                },
            },
            "search": {"repetition_penalty": 1.1},
        }
        (self.model / "model.onnx").write_bytes(b"model")
        self.write_config()

    def write_config(self):
        (self.model / "genai_config.json").write_text(json.dumps(self.config), encoding="utf-8")

    def generate_config(self, provider, mode, fpa_intb=False):
        path = benchmark.create_benchmark_config(
            self.model, provider, mode, 4096, 128, fpa_intb, 0, self.root / f"{provider}-{mode}"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_webgpu_replaces_cuda_and_bounds_generation(self):
        config = self.generate_config("webgpu", "planned")
        session = config["model"]["decoder"]["session_options"]
        self.assertEqual(session["provider_options"], [{"webgpu": {}}])
        self.assertFalse(any(key.startswith("ep.cuda.") for key in session))
        self.assertEqual(session["session.enable_static_workspace_preallocation"], "1")
        self.assertEqual(
            session["session.max_shape_override"],
            "input_ids:[1,4096];attention_mask:[1,4224];"
            "past_key_values.0.key:[1,2,4224,8];past_key_values.0.value:[1,2,4224,8];"
            "past_key_values.1.key:[1,2,4224,8];past_key_values.1.value:[1,2,4224,8]",
        )
        self.assertEqual(config["search"]["max_length"], 4224)
        self.assertEqual(config["search"]["min_length"], 4224)
        self.assertFalse(config["search"]["do_sample"])
        self.assertTrue(config["search"]["past_present_share_buffer"])
        self.assertEqual(config["search"]["repetition_penalty"], 1.1)
        self.assertEqual(json.loads((self.model / "genai_config.json").read_text()), self.config)
        self.assertEqual((self.root / "webgpu-planned" / "model.onnx").read_bytes(), b"model")

    def test_webgpu_preserves_existing_provider_options(self):
        self.config["model"]["decoder"]["session_options"]["provider_options"].append(
            {"WebGPU": {"dawnBackendType": "Vulkan"}}
        )
        self.write_config()
        config = self.generate_config("webgpu", "scratch")
        session = config["model"]["decoder"]["session_options"]
        self.assertEqual(session["provider_options"], [{"webgpu": {"dawnBackendType": "Vulkan"}}])
        self.assertEqual(session["session.enable_static_workspace_preallocation"], "0")

    def test_cuda_retains_provider_options_and_modes(self):
        for mode in benchmark.CUDA_MODES:
            with self.subTest(mode=mode):
                config = self.generate_config("cuda", mode, fpa_intb=True)
                session = config["model"]["decoder"]["session_options"]
                self.assertEqual(
                    session["provider_options"],
                    [{"cuda": {"enable_cuda_graph": "0", "enable_skip_layer_norm_strict_mode": "1", "device_id": "0"}}],
                )
                self.assertEqual(session["ep.cuda.fpa_intb_gemm"], "1")
                self.assertEqual(
                    session["session.enable_static_workspace_preallocation"], "0" if mode == "scratch" else "1"
                )
                if mode == "combined":
                    self.assertEqual(session["ep.cuda.gqa_workspace_max_total_sequence_length"], "4224")
                else:
                    self.assertNotIn("ep.cuda.gqa_workspace_max_total_sequence_length", session)

    def test_invalid_provider_mode_and_capacity_fail(self):
        for provider, mode, fpa in [
            ("invalid", "scratch", False),
            ("webgpu", "combined", False),
            ("webgpu", "planned", True),
            ("cuda", "planned", False),
        ]:
            with self.subTest(provider=provider, mode=mode, fpa=fpa), self.assertRaises(ValueError):
                self.generate_config(provider, mode, fpa)
        self.config["model"]["context_length"] = 4096
        self.write_config()
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.generate_config("webgpu", "scratch")

    def test_controlled_provider_options_change_only_requested_factors(self):
        for cache in ("disabled", "bucket"):
            for mode in benchmark.WEBGPU_MODES:
                with self.subTest(cache=cache, mode=mode):
                    path = benchmark.create_benchmark_config(
                        self.model,
                        "webgpu",
                        mode,
                        8192,
                        128,
                        False,
                        0,
                        self.root / f"controlled-{cache}-{mode}",
                        cache,
                        True,
                    )
                    config = json.loads(path.read_text())
                    session = config["model"]["decoder"]["session_options"]
                    self.assertEqual(
                        session["provider_options"],
                        [
                            {
                                "webgpu": {
                                    "storageBufferCacheMode": cache,
                                    "enableInt64": "1",
                                    "enableGraphCapture": "0",
                                    "preferredLayout": "NHWC",
                                }
                            }
                        ],
                    )
                    self.assertEqual(
                        session["session.enable_static_workspace_preallocation"], "0" if mode == "scratch" else "1"
                    )
                    self.assertEqual(config["search"]["max_length"], 8320)

    def test_memory_windows_exclude_warmup_and_keep_whole_process_peak(self):
        sampler = benchmark.NvidiaSmiMemorySampler(0, 5)
        sampler.samples = [(100, 0), (200, 2000), (300, 500), (400, 700), (500, 0)]
        whole = sampler.summarize_window()
        steady = sampler.summarize_window(250, 450)
        self.assertEqual(whole["device_peak_mib"], 2000)
        self.assertEqual(steady["device_baseline_mib"], 500)
        self.assertEqual(steady["device_peak_mib"], 700)
        self.assertEqual(steady["sample_count"], 2)
        with self.assertRaisesRegex(RuntimeError, "No nvidia-smi samples"):
            sampler.summarize_window(210, 290)

    def test_prompt_matches_across_providers(self):
        prompt = benchmark.make_prompt_tokens(self.config, 1024, 1234)
        np.testing.assert_array_equal(prompt, benchmark.make_prompt_tokens(self.config, 1024, 1234))
        self.assertEqual(prompt[0], 1)
        self.assertFalse(np.isin(prompt[1:], [0, 1, 2, 3]).any())

    def test_worker_command_preserves_provider_and_token_count(self):
        with patch(
            "sys.argv",
            ["benchmark", "--model-path", str(self.model), "--execution-provider", "webgpu", "--phase", "scenario"],
        ):
            args = benchmark.parse_arguments()
        command = benchmark.worker_command(args, "planned", self.root / "worker.json")
        self.assertEqual(command[command.index("--execution-provider") + 1], "webgpu")
        self.assertEqual(command[command.index("--mode") + 1], "planned")
        self.assertEqual(command[command.index("--generated-tokens") + 1], "128")

    def test_cli_rejects_inapplicable_webgpu_flags(self):
        for flags in [["--fpa-intb"], ["--mode", "combined"], ["--device-id", "1"]]:
            with (
                self.subTest(flags=flags),
                patch(
                    "sys.argv",
                    ["benchmark", "--model-path", str(self.model), "--execution-provider", "webgpu", *flags],
                ),
                self.assertRaises(SystemExit),
                patch("sys.stderr"),
            ):
                benchmark.parse_arguments()

    def test_paired_statistics_match_cuda_definitions(self):
        results = [
            {"mode": "scratch", "block_index": 0, "request_ttft_ms": {"count": 3, "trimmed_mean": 100}},
            {"mode": "planned", "block_index": 0, "request_ttft_ms": {"count": 3, "trimmed_mean": 90}},
            {"mode": "planned", "block_index": 1, "request_ttft_ms": {"count": 3, "trimmed_mean": 110}},
            {"mode": "scratch", "block_index": 1, "request_ttft_ms": {"count": 3, "trimmed_mean": 100}},
        ]
        changes = benchmark.summarize_paired_changes(results, "request_ttft_ms", "planned", "scratch")
        self.assertAlmostEqual(changes["changes_by_block_pct"][0], -10)
        self.assertAlmostEqual(changes["changes_by_block_pct"][1], 10)
        self.assertAlmostEqual(changes["change_pct"]["median"], 0)

    def test_generator_requires_exact_token_and_decode_counts(self):
        prompt = np.array([1, 4, 5], dtype=np.int32)
        for output_length, decode_count, valid in [(6, 2, True), (4, 0, False), (6, 3, False)]:
            with self.subTest(output_length=output_length, decode_count=decode_count):
                og = Mock()
                generator = og.Generator.return_value
                generator.get_sequence.return_value = list(range(output_length))
                generator.is_done.side_effect = [False] * decode_count + [True]
                if valid:
                    result = benchmark.run_generation(og, object(), prompt, 3)
                    self.assertEqual(result["output_length"], 6)
                    self.assertEqual(len(result["decode_ms"]), 2)
                    self.assertEqual(generator.get_next_tokens.call_count, 3)
                else:
                    with self.assertRaisesRegex(RuntimeError, "Expected 3 generated tokens"):
                        benchmark.run_generation(og, object(), prompt, 3)

    def test_engine_rejects_short_generation_and_closes_request(self):
        og = Mock()
        og.EngineEventFlags.FAILED = 1
        og.EngineEventFlags.TOKEN = 2
        engine = Mock()
        request = engine.create_request.return_value
        engine.has_pending_requests.side_effect = [True, False]
        engine.run.return_value = [Mock(flags=2, request=request, token=4)]
        with self.assertRaisesRegex(RuntimeError, "Expected 3 generated tokens, got 1"):
            benchmark.run_engine_generation(og, engine, np.array([1], dtype=np.int32), 3)
        request.close.assert_called_once()

    def test_controller_preserves_modes_order_and_report_schema(self):
        metrics = (
            "generator_setup_ms",
            "append_tokens_ms",
            "sampling_ms",
            "request_ttft_ms",
            "model_ttft_ms",
            "request_scenario_ms",
            "model_scenario_ms",
            "decode_total_ms",
            "decode_token_ms",
        )
        for provider in ("cuda", "webgpu"):
            for phase in ("scenario", "memory"):
                with (
                    self.subTest(provider=provider, phase=phase),
                    patch(
                        "sys.argv",
                        [
                            "benchmark",
                            "--model-path",
                            str(self.model),
                            "--execution-provider",
                            provider,
                            "--phase",
                            phase,
                            "--repetitions",
                            "2",
                            "--output",
                            str(self.root / f"{provider}-{phase}.json"),
                        ],
                    ),
                ):
                    args = benchmark.parse_arguments()
                    modes = []

                    def run_fake_worker(command, check, provider=provider, modes=modes):
                        self.assertTrue(check)
                        self.assertEqual(command[command.index("--execution-provider") + 1], provider)
                        mode = command[command.index("--mode") + 1]
                        modes.append(mode)
                        result = {
                            "mode": mode,
                            "output_hashes": ["same-output"],
                            "output_lengths": [1152],
                            "memory": {
                                "device_baseline_mib": 100,
                                "device_peak_mib": 200,
                                "device_peak_delta_mib": 100,
                            },
                        }
                        result.update({metric: benchmark.summarize([10.0, 11.0, 12.0]) for metric in metrics})
                        Path(command[command.index("--worker-output") + 1]).write_text(
                            json.dumps(result), encoding="utf-8"
                        )

                    with (
                        patch.object(benchmark.subprocess, "run", side_effect=run_fake_worker),
                        patch("builtins.print"),
                    ):
                        report = benchmark.run_controller(args)
                    expected_modes = (
                        ["scratch", "matmul", "combined", "scratch", "combined", "matmul"]
                        if provider == "cuda"
                        else ["scratch", "planned", "planned", "scratch"]
                    )
                    self.assertEqual(modes, expected_modes)
                    self.assertTrue(report["output_validation"]["outputs_match"])
                    self.assertEqual(set(report["aggregate"]), set(benchmark.modes_for_provider(provider)))
                    self.assertEqual(json.loads(Path(args.output).read_text()), report)
                    comparison_keys = (
                        {"matmul_vs_scratch", "combined_vs_scratch", "combined_vs_matmul"}
                        if provider == "cuda"
                        else {"planned_vs_scratch"}
                    )
                    if phase == "memory":
                        self.assertEqual(set(report["paired_memory"]), comparison_keys)
                        self.assertIsNone(report["paired_changes"])
                    else:
                        self.assertEqual(set(report["paired_changes"]["request_ttft_ms"]), comparison_keys)
                        self.assertIsNone(report["paired_memory"])


if __name__ == "__main__":
    unittest.main()
