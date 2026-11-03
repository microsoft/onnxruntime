import unittest

from prepare_dense_long_k_model_v1 import ROOT, model_source


class LongKModelTests(unittest.TestCase):
    def test_original_controls_exact_late_gpu_routing_and_natural_thresholds(self):
        source = model_source((ROOT / "run_context_flag_cache_model.py").read_text())
        scope = {"__name__": "long_k_model_test"}
        exec(compile(source, "long-k-model-pipeline", "exec"), scope)
        for flag in (0, 1):
            for trace in (0, 1):
                prefix = scope["prefix_for"](2048, True, flag, 0, trace)
                shell = scope["command_for"](prefix, 2048, True, flag, trace)[-1]
                expected = "qwen38-int4-dense-tile-parallel3-long-k" if flag else "qwen38-int4-fc2-private-a"
                self.assertIn(f"/workspace/{expected}/build-v1/libonnxruntime_providers_webgpu.so", shell)
                self.assertIn("--webgpu-validation-mode full", shell)
                self.assertIn("benchmark_dense_long_k.py", shell)
                self.assertIn("--tokens 36 --repetitions 1 --profile-decode-after 35" if trace
                              else "--tokens 256 --repetitions 3", shell)
        event = {"cat": "Kernel", "name": "node&" + scope["NAME"], "pid": -1, "tid": -1,
                 "ph": "X", "dur": 16, "args": {"cache_key": "program:384,1,1:shape"}}
        programs = scope["model_programs"]
        self.assertEqual(programs([event] * 48, 1), {"DenseColumns8DoubleBuffer": 48})
        for events, flag in (([event] * 47, 1), ([event] * 48, 0),
                             ([dict(event, dur=float("nan"))] * 48, 1)):
            with self.assertRaises(RuntimeError):
                programs(events, flag)
        self.assertIn('raw["plugin_sha256"] != expected_provider_sha', source)
        self.assertIn('routing[0]["normalized_program_counts"] != routing[1]["normalized_program_counts"]', source)
        self.assertIn("compare(groups[1], groups[0], groups[2])", source)
        self.assertIn('"held_no_unchanged_retry"', source)
        self.assertIn('sample["prefill_timing_scope"] != "append_tokens_through_first_token_selection"', source)


if __name__ == "__main__":
    unittest.main()
