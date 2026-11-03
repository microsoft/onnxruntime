from pathlib import Path
import unittest

from build_qmoe_fc2_final_mix import FLAG, FUSED, transform


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-fc1-activation/build-v3/overlay/"
              "contrib_ops/webgpu/moe/qmoe.cc")


class QmoeFC2FinalMixTests(unittest.TestCase):
    def test_scoped_rewrite_and_capture_count(self):
        source = SOURCE.read_text()
        result = transform(source)
        self.assertIn(f'"{FLAG} must be 0 or 1"', result)
        self.assertIn("key.push_back(fc2_mix)", result)
        self.assertIn("(CanFuseFc2FinalMix(context, fc2_mix == 1) ? 1 : 0)", result)
        self.assertEqual(result.count(FUSED), 1)
        start = source.index("    status = is_nvfp4_ || use_int4_cooperative",
                             source.index("    Tensor& fc2_outputs = temporary(4"))
        end = source.index("\n    return Status::OK();", start)
        self.assertIn(source[start:end], result)
        self.assertEqual(source[end:], result[result.index("\n    return Status::OK();", start):])

    def test_cell_ownership_and_bounds(self):
        cells = {(thread // 32 * 2 + c, thread % 32) for thread in range(320) for c in range(2)}
        self.assertEqual(cells, {(cell, lane) for cell in range(20) for lane in range(32)})
        reads = {rank * 160 + lane * 8 + i * 2 + v
                 for rank in range(10) for lane in range(20) for i in range(4) for v in range(2)}
        self.assertEqual(reads, set(range(1600)))

    def test_fp16_materialization_and_ordered_mix(self):
        source = (ROOT / "qmoe_fc2_final_mix.inc").read_text()
        self.assertIn("partials[local_idx][0] = value", source)
        self.assertIn("var acc = output_element_t(0)", source)
        self.assertIn("r < uniforms.k; r++", source)
        self.assertIn("acc += router_value * partials[r * 2u + local_idx][0]", source)
        self.assertNotIn("f32(", source)
        self.assertIn("maxComputeWorkgroupStorageSize >= 14080", source)
        self.assertIn("context.Input<Tensor>(12) == nullptr", source)


if __name__ == "__main__":
    unittest.main()
