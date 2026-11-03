import json
import unittest

import benchmark_bind_group_inline_key as wrapper
from benchmark_gpu_slice import run_request
from prepare_webgpu_int4_metrics import ROOT
from run_bind_group_inline_key_model import command_for, prefix_for, verify_routing


class BindGroupInlineKeyModelTests(unittest.TestCase):
    def test_request_implementation_is_reused_unchanged(self):
        self.assertIs(wrapper.run_request, run_request)

    def test_scope_and_flags_for_full_and_retained(self):
        for context in (2048, 8192):
            for full in (True, False):
                for flag in (0, 1):
                    command = command_for(prefix_for(context, full, flag, 0, 0), context, full, flag, 0)
                    self.assertEqual(command[:3], ["taskset", "-c", "16-23"])
                    shell = command[-1]
                    self.assertIn(f"ORT_WEBGPU_BIND_GROUP_INLINE_KEY={flag}", shell)
                    self.assertIn("ORT_WEBGPU_QMOE_FC2_PRIVATE_A=0", shell)
                    self.assertIn("ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH=0", shell)
                    self.assertIn("ORT_GENAI_WEBGPU_GPU_SLICE_COPY=1", shell)
                    self.assertIn(f"--contexts {context}", shell)
                    self.assertEqual("--webgpu-validation-mode full" in shell, full)
                    self.assertIn("--tokens 256 --repetitions 3", shell)
                    self.assertIn("--inherit-worker-cpu-affinity", shell)

    def test_routing_validator_accepts_actual_qualified_ordinary_trace(self):
        prefix = ROOT / "gpu-fused-slice-v1-context2048-full-flag0-trial0-trace1-gpu1"
        verify_routing(json.loads(prefix.with_suffix(".json").read_text()), prefix.with_suffix(".log").read_text(), 1)

    def test_routing_rejects_fallback_and_trace_during_natural_timing(self):
        raw = {"samples": [{"output_tokens": 1}]}
        log = ("WEBGPU_GREEDY_ARGMAX rows=1\nWEBGPU_GREEDY_MIN_LENGTH rows=1\n"
               "WEBGPU_GPU_SLICE_COPY source_bytes=508559360 offset_bytes=508062720 copied_bytes=496640\n")
        verify_routing(raw, log, 1)
        with self.assertRaises(RuntimeError):
            verify_routing(raw, log + "WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK\n", 1)
        with self.assertRaises(RuntimeError):
            verify_routing(raw, log, 0)


if __name__ == "__main__":
    unittest.main()
