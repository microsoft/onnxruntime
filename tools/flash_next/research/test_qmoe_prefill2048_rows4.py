from pathlib import Path
import math
import unittest

from build_prefill2048_rows4_dsa import ANCHOR, DSA_EXTENSION, transform as transform_dsa
from build_qmoe_prefill2048_rows4 import (
    FALLBACK_ROWS, FLAG, PROGRAM, ROWS, transform_compare, transform_matmul, transform_qmoe)
from qualify_quickgelu_mul_long_context import ROOT
from run_prefill2048_rows4_model import model_command, verify_flags, verify_large_profile
from run_qmoe_grouped_prefill_native import PROGRAMS
from run_qmoe_prefill_rows4_standard import TEMPLATE
import json


REPOS = Path("/home/sayanshaw/repos")
GROUPED = REPOS / "qwen38-int4-qmoe-grouped-prefill/build-v1"
PARENT = REPOS / "qwen38-int4-qmoe-prefill-rows4/build-v2"


class Prefill2048Rows4Tests(unittest.TestCase):
    def test_qmoe_new_flag_is_validated_and_replay_keyed(self):
        actual = transform_qmoe((GROUPED / "qmoe.cc").read_text())
        self.assertEqual(actual.count(f'"{FLAG}"'), 2)
        self.assertEqual(actual.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(prefill2048_rows4)", actual)
        self.assertIn("moe_params.num_rows == 2048", actual)
        self.assertIn("grouped_prefill == 1 && use_extended_int4_prefill", actual)
        self.assertIn("maxComputeWorkgroupsPerDimension >= 20480", actual)
        self.assertIn("use_prefill2048_rows4 ? 20480 : use_prefill1024 ? 10240", actual)
        self.assertIn("(use_prefill512 || use_prefill1024 || use_prefill2048_rows4)", actual)

    def test_large_activation_flat_dispatch_has_exact_coverage(self):
        context = (REPOS / "qwen38-int4-static-node-replay/build-v8/overlay/"
                   "core/providers/webgpu/webgpu_context.cc").read_text()
        self.assertIn("program_mgr_->NormalizeDispatchGroupSize(x, y, z)", context)
        core = REPOS / "onnxruntime-webgpu-h200-perf/onnxruntime/core/providers/webgpu"
        manager = (core / "program_manager.cc").read_text()
        self.assertIn("std::ceil(std::sqrt(static_cast<double>(size)))", manager)
        self.assertIn("x = y = side;\n      z = 1;", manager)
        helper = (core / "shader_helper.cc").read_text()
        self.assertIn("workgroup_id.z * num_workgroups[0] * num_workgroups[1] + "
                      "workgroup_id.y * num_workgroups[0] + workgroup_id.x", helper)
        self.assertIn("workgroup_idx * (workgroup_size_x * workgroup_size_y * workgroup_size_z) + local_idx", helper)
        logical = 2048 * 10 * 640 // 128
        self.assertEqual(logical, 102400)
        side = math.ceil(math.sqrt(logical))
        self.assertEqual((side, side, 1), (320, 320, 1))
        self.assertLessEqual(side, 65535)
        self.assertEqual([y * side + x for y in range(side) for x in range(side)],
                         list(range(logical)))

    def test_attention_extension_changes_only_scoped_admission(self):
        source = (REPOS / "qwen38-int4-dsa-query-cache/build-v2/dynamic_sparse_attention.cc").read_text()
        actual = transform_dsa(source)
        self.assertEqual(actual.replace(DSA_EXTENSION, ANCHOR), source)
        self.assertIn(f'"{FLAG}"', actual)
        self.assertIn(f"{FLAG} must be 0 or 1", actual)
        self.assertIn("parameters.batch_size == 1 && parameters.sequence_length == 2048", actual)
        self.assertLessEqual(2048 * 24, 65535)

    def test_shader_arithmetic_and_old_rows_are_unchanged(self):
        original = (PARENT / "matmul_nbits.cc").read_text()
        actual = transform_matmul(original)
        start = original.index("WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(", original.index("class GroupedPrefillRows4Program"))
        end = original.index("class MatMulInt3Program", start)
        self.assertIn(original[start:end], actual)
        self.assertIn(f'large ? "{PROGRAM}" : "MatMulNBitsGroupedRows4F16"', actual)
        self.assertIn("M == 5120 || M == 10240 || (M == 20480 && prefill2048_rows4 == 1)", actual)
        self.assertIn("GroupedPrefillRows4Program program{has_bias, M == 20480}", actual)

    def test_comparator_covers_large_and_neighbor_fallbacks(self):
        actual = transform_compare((GROUPED / "native_compare.cpp").read_text())
        self.assertIn("1025, 2047, 2048, 2049", actual)
        self.assertEqual(actual.count("rows > 512 && rows != 1024 && rows != 2048"), 2)
        self.assertIn("last_router.data(), 2049", actual)
        self.assertEqual(6 * 2560 * sum(r * (2 if r in FALLBACK_ROWS else 8) for r in ROWS), 678297600)

    def test_commands_preserve_template_and_separate_chunk_controls(self):
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        original = template.copy()
        prefix = ROOT / "prefill2048-rows4-command-unit-test"
        for control in (False, True):
            for full in (False, True):
                command = model_command(template, PARENT / "libonnxruntime_providers_webgpu.so",
                                        prefix, 2048, full, control)
                self.assertIn(f"--chunk {1024 if control else 2048}", command[-1])
                self.assertIn(f"{FLAG}={int(not control)}", command[-1])
                self.assertIn("--tokens 256 --repetitions 3", command[-1])
                self.assertEqual(" --webgpu-validation-mode full" in command[-1], full)
                self.assertNotIn("--profile-prefix", command[-1])
        self.assertEqual(template, original)
        profile = model_command(template, PARENT / "libonnxruntime_providers_webgpu.so",
                                prefix, 2048, True, profile=True)
        self.assertIn("--tokens 2 --repetitions 1", profile[-1])
        self.assertIn(f"--profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile", profile[-1])
        with self.assertRaises(AssertionError):
            model_command(template, PARENT / "libonnxruntime_providers_webgpu.so",
                          prefix, 2048, False, profile=True)

    def test_new_selector_required_without_enabling_old_2048_route(self):
        required = ("ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_QSA_SCORE16",
                    "ORT_WEBGPU_QMOE_PREFILL512", "ORT_WEBGPU_QMOE_PREFILL1024",
                    "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32",
                    "ORT_WEBGPU_QMOE_GROUPED_PREFILL", "ORT_WEBGPU_QSA_PREFILL_SCORE16", FLAG)
        flags = {name: "1" for name in required}
        flags.update(ORT_WEBGPU_QMOE_PREFILL2048="0", ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD="0")
        verify_flags({"load_experiment_flags": flags})
        for name in required:
            with self.assertRaises(AssertionError):
                verify_flags({"load_experiment_flags": dict(flags, **{name: "0"})})
        with self.assertRaises(AssertionError):
            verify_flags({"load_experiment_flags": dict(flags, ORT_WEBGPU_QMOE_PREFILL2048="1")})

    def test_actual_profile_requires_all_large_rows_and_both_projections(self):
        events = [{"cat": "Kernel", "name": f"node&{name}", "args": {}}
                  for name in PROGRAMS for _ in range(96)]
        events += [{"cat": "Kernel", "name": "node&QmoeGate1Token",
                    "args": {"shapes": "inputs[0] = {1,2048,512}"}} for _ in range(96)]
        for width, columns in ((640, 1280), (160, 2560)):
            events += [{"cat": "Kernel", "name": f"node&{PROGRAM}",
                        "args": {"shapes": f"inputs[0] = {{20480,{width}}} "
                                          f"outputs[0] = {{20480,{columns}}}"}} for _ in range(96)]
        self.assertEqual(verify_large_profile(events)[PROGRAM], 192)
        with self.assertRaises(AssertionError):
            verify_large_profile(events[:-1])
        malformed = [dict(event) for event in events]
        malformed[-1]["args"] = {"shapes": "inputs[0] = {10240,160} outputs[0] = {10240,2560}"}
        with self.assertRaises(AssertionError):
            verify_large_profile(malformed)


if __name__ == "__main__":
    unittest.main()
