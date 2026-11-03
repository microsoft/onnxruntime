from pathlib import Path
import unittest
from unittest.mock import patch

from build_dense_lane8 import DISPATCH, SELECTOR, lane8_template, private_parts, transform
from run_dense_lane8_native import main as run_native


ROOT = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32")
PARENT = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2")


class DenseLane8Tests(unittest.TestCase):
    def test_bijection_and_warp_geometry(self):
        lanes = [(lane % 4) * 32 + (lane // 4) % 8 + (lane // 32) * 8
                 for lane in range(128)]
        self.assertEqual(sorted(lanes), list(range(128)))
        for warp in range(4):
            logical = lanes[warp * 32:(warp + 1) * 32]
            for output in range(4):
                self.assertEqual(sorted(lane % 32 for lane in logical if lane // 32 == output),
                                 list(range(warp * 8, warp * 8 + 8)))

    def test_template_only_changes_lane_mapping(self):
        original = (ROOT / "block32-template-v1/dense_block32_template.inc").read_text()
        modified = lane8_template(original)
        declaration = next(line for line in modified.splitlines()
                           if line.startswith('ss << "  let logical_lane = '))
        restored = modified.replace(declaration + "\n", "").replace(
            "logical_lane % ", "local_idx % ").replace("logical_lane / ", "local_idx / ").replace(
                "ApplyDenseLane8Template", "ApplyDenseBlock32Template")
        self.assertEqual(restored, original)

    def test_exact_reversible_kernel_delta(self):
        original = (PARENT / "matmul_nbits.cc").read_text()
        program, function = private_parts(original)
        modified = transform(original)
        restored = modified.replace('#include "dense_lane8_template.inc"\n', "").replace(
            "\n\n" + program, "", 1).replace("\n\n" + function, "", 1).replace(
                SELECTOR, "", 1).replace(DISPATCH, "", 1).replace(
                    "(block32_mode == 1 ? (dense_lane8 ? 4 : 3) : 1 + dense_tile4)",
                    "(block32_mode == 1 ? 3 : 1 + dense_tile4)")
        self.assertEqual(restored, original)
        self.assertIn("class GroupedPrefillRows4Program", modified)
        self.assertIn('Program{"MatMulNBitsLane8F16"}', program)
        self.assertIn('vendor == std::string_view{"nvidia"}', SELECTOR)
        self.assertIn("!context.EnableMatmulFp32Accumulation()", SELECTOR)
        self.assertIn("dense_bitcast_route && block32_mode == 1 && dense_lane8", DISPATCH)

    def test_native_and_fp32_fallback_wiring(self):
        for fp32 in (False, True):
            argv = ["run_dense_lane8_native.py"] + (["--fp32-accumulation"] if fp32 else [])
            with self.subTest(fp32=fp32), patch("sys.argv", argv), patch(
                    "run_dense_lane8_native.run_native") as runner:
                run_native()
                kwargs = runner.call_args.kwargs
                self.assertEqual(kwargs["fp32_accumulation"], fp32)
                self.assertEqual(kwargs["expect_replay"], not fp32)
                self.assertIn("ORT_WEBGPU_DENSE_BLOCK32=1 ", kwargs["extra_flags"])
                self.assertIn(f"ORT_TEST_DENSE_FP32_ACCUMULATION={int(fp32)} ", kwargs["extra_flags"])
                self.assertIn("qmoe-prefill-rows4-v2-contexts", kwargs["baseline_path"].name)
                self.assertEqual(kwargs["directory"].name, "build-v2")


if __name__ == "__main__":
    unittest.main()
