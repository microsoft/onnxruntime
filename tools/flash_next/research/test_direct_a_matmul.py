from pathlib import Path
import unittest

from build_direct_a_matmul import FLAG, ROUTE, transform_helper, transform_matmul, transform_qmoe


ROOT = Path(__file__).parent
FROZEN = Path("/home/sayanshaw/repos/qwen38-int4-static-node-replay/build-v8")


class DirectAMatMulTests(unittest.TestCase):
    def test_shader_keeps_order_and_only_final_barrier(self):
        source = (ROOT / "direct_a_matmul.inc").read_text()
        self.assertEqual(source.count("workgroupBarrier();"), 1)
        self.assertNotIn("tile_A", source)
        self.assertIn("tile += 32u", source)
        self.assertIn("offset += 4u", source)
        self.assertIn("sum += dot(x0, w0) + dot(x1, w1)", source)
        self.assertIn("partials[offset + group][lane] += sum", source)
        self.assertIn("i < 32u; i++) { value += partials[local_idx][i]", source)
        self.assertIn("value += bias[expert * uniforms.N + col]", source)

    def test_unique_partial_ownership(self):
        owners = [(offset + lane // 32, lane % 32)
                  for lane in range(128) for offset in (0, 4)]
        self.assertEqual(len(set(owners)), 256)
        self.assertEqual(set(owners), {(column, lane) for column in range(8) for lane in range(32)})

    def test_conservative_route(self):
        for guard in ("batch_count == 1", "M == 1", "has_weight_idx_indirect && dispatch_M == 10", "nbits == 4",
                      "block_size == 32", "K % 32 == 0", "!has_zero_points",
                      "!single_scale_weights", "components_a == 4", "components_b == 4",
                      "!acc_f32", 'std::string_view{"nvidia"}'):
            self.assertIn(guard, ROUTE)

    def test_leaf_cache_variant(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-generic-matmul-replay/build-v1/matmul_nbits.cc").read_text()
        result = transform_matmul(source)
        self.assertEqual(result.count(f'const int direct_a = ParseEnvironmentVariableWithDefault<int>("{FLAG}"'), 2)
        self.assertIn("static_replay_.Run(context, {y}, compute, 0, direct_a)", result)
        self.assertLess(result.index("if (use_wide_tile_program)"), result.index(ROUTE))
        helper = transform_helper((FROZEN / "overlay/static_node_replay.h").read_text())
        self.assertIn("uint64_t shader_variant = 0", helper)
        self.assertIn("key.push_back(shader_variant)", helper)

    def test_qmoe_cache_variant(self):
        source = (ROOT / "qmoe_node_replay.inc").read_text()
        result = transform_qmoe(source)
        self.assertLess(result.index("direct_a == 0 || direct_a == 1"), result.index("if (enabled == 0)"))
        self.assertIn("key.push_back(direct_a)", result)


if __name__ == "__main__":
    unittest.main()
