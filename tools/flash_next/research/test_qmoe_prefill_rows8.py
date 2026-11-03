import json
from pathlib import Path
import unittest

from build_qmoe_prefill_rows8 import FLAG, transform, transform_program, transform_replay
from run_qmoe_prefill256_native import verify_additional_dispatches
from run_qmoe_prefill_rows8_model import command_for_case, verify_profile
from run_qmoe_grouped_prefill_native import PROGRAMS


class PrefillRows8Tests(unittest.TestCase):
    def test_activation_loads_and_output_rows_exactly_cover_admitted_shapes(self):
        loads = [index for thread in range(128) for index in range(thread, 2048, 128)]
        self.assertEqual(sorted(loads), list(range(2048)))
        for dimension in (640, 2560):
            for kidx in range(0, dimension, 1024):
                valid = [(index // 256, kidx // 4 + index % 256) for index in loads
                         if kidx // 4 + index % 256 < dimension // 4]
                self.assertEqual(sorted(valid), [(row, k) for row in range(8)
                                                 for k in range(kidx // 4, min(kidx // 4 + 256, dimension // 4))])
        for rows in (5120, 10240):
            self.assertEqual([group * 8 + row for group in range(rows // 8) for row in range(8)],
                             list(range(rows)))

    def test_original_dot_order_typed_updates_and_serial_fold_preserved(self):
        source = Path(__file__).with_name("qmoe_prefill_rows4.inc").read_text()
        actual = transform_program(source)
        for expression in ("partials[row][slot + idy][idx] += sum;",
                           "value += partials[row][local_idx][lane];",
                           "expert != cached_expert", "kidx += 1024u",
                           "sum += dot(tile_a[offset], weights[word * 2u])"):
            self.assertIn(expression, actual)
        self.assertEqual(actual.count("workgroupBarrier();"), 2)
        self.assertIn("row < 8u", actual)
        self.assertNotIn("row < 4u", actual)
        self.assertNotIn("subgroupMatrix", actual)

    def test_original_route_preserved_limits_and_replay_keyed(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc").read_text()
        actual = transform(source)
        start = source.index("  if (grouped_prefill_rows == 1", source.index("Status ApplyMatMulNBits("))
        self.assertIn(source[start:], actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 20512", actual)
        self.assertIn("N / 8, M / 8", actual)
        replay = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        replay = transform_replay(replay)
        self.assertEqual(replay.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(prefill_rows8)", replay)

    def test_native_profiles_cover_only_candidate_measured_session(self):
        expected = {**{name: 16 for name in PROGRAMS}, "MatMulNBitsGroupedRows8F16": 32}
        events = [{"cat": "Kernel", "name": "QMoE&" + name}
                  for name, count in expected.items() for _ in range(count)]
        self.assertEqual(verify_additional_dispatches(events, expected), expected)
        with self.assertRaises(AssertionError):
            verify_additional_dispatches(events, {**expected, "MatMulNBitsGroupedRows4F16": 32})

    def test_both_model_projections_and_absent_original_required(self):
        events = [{"cat": "Kernel", "name": "QMoE&" + name} for name in PROGRAMS for _ in range(192)]
        for width, columns in ((640, 1280), (160, 2560)):
            events += [{"cat": "Kernel", "name": "QMoE&MatMulNBitsGroupedRows8F16",
                        "args": {"shapes": f"inputs[0] = {{10240,{width}}} outputs[0] = {{10240,{columns}}}"}}
                       for _ in range(192)]
        self.assertEqual(verify_profile(events, 2048)["active_rows8_prefill_projections"], 384)
        with self.assertRaises(AssertionError):
            verify_profile(events[:-1], 2048)
        with self.assertRaises(AssertionError):
            verify_profile(events + [{"cat": "Kernel", "name": "QMoE&MatMulNBitsGroupedRows4F16"}], 2048)

    def test_screen_commands_keep_grouping_and_toggle_only_rows8(self):
        root = Path(__file__).parent
        baseline = "dense-block32-v2-chunk1024-gpu7-full-standard2048-20261030"
        template = json.loads((root / f"{baseline}.guard.json").read_text())["command"]
        for full in (True, False):
            for mode in (0, 1):
                command = command_for_case(template, baseline, root / "rows8-test",
                                           Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows8/build-v1/"
                                                "libonnxruntime_providers_webgpu.so"),
                                           mode, False, full)
                self.assertIn(f"{FLAG}={mode}", command[-1])
                self.assertIn("ORT_WEBGPU_QMOE_GROUPED_PREFILL=1", command[-1])
                self.assertIn("--tokens 36", command[-1])
                self.assertEqual(" --webgpu-validation-mode full" in command[-1], full)
                self.assertNotIn("--profile-prefix", command[-1])


if __name__ == "__main__":
    unittest.main()
