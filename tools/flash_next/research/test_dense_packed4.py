from pathlib import Path
import unittest

import build_dense_packed4
from build_dense_packed4 import PACK, REPLAY, packed_replay, packed_template, transform
from run_dense_packed4_native import verify_packed_stats


PARENT = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2")
HEADER = Path("/home/sayanshaw/repos/qwen38-int4-static-node-replay/build-v8/overlay/"
              "contrib_ops/webgpu/quantization/matmul_nbits.h")
TEMPLATE = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/block32-template-v1/dense_block32_template.inc")


class DensePacked4Tests(unittest.TestCase):
    def test_manifest_include_paths_follow_build_version(self):
        source = Path(build_dense_packed4.__file__).read_text()
        self.assertIn('f"../{INCLUDES.name}/{item.name}"', source)
        self.assertNotIn('"../packed4-includes-v1/"', source)

    def test_producer_avoids_reserved_target_identifier(self):
        self.assertNotIn("let target =", PACK)
        self.assertIn('SetByOffset("packed_offset"', PACK)

    def test_cached_bindings_match_shader_weight_activation_scale_order(self):
        original = Path(str(REPLAY).replace("/workspace/", "/home/sayanshaw/repos/"))
        replay = packed_replay(original.read_text())
        markers = ["append_extra(*extra_inputs.begin());", "for (int index : gpu_inputs_)",
                   "append_extra(*(extra_inputs.begin() + 1));", "for (auto* tensor : outputs)"]
        offsets = [replay.index(marker) for marker in markers]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("extra_inputs.size() == 2 && gpu_inputs_.size() == 1 && gpu_inputs_[0] == 0", replay)
        self.assertIn("entry.bindings.buffers == expected", replay)

    def test_complete_byte_block_and_scale_bijection(self):
        for columns in (4, 8, 12):
            for blocks in (1, 2, 3, 80):
                with self.subTest(columns=columns, blocks=blocks):
                    targets = [((column // 4) * blocks + block) * 4 + column % 4
                               for column in range(columns) for block in range(blocks)]
                    self.assertEqual(sorted(targets), list(range(columns * blocks)))
                    weights = [bytes((index * 17 + byte) % 256 for byte in range(16))
                               for index in range(columns * blocks)]
                    scales = [(index * 319 + 17) % 65536 for index in range(columns * blocks)]
                    packed, packed_scales = [None] * len(weights), [None] * len(scales)
                    for index, target in enumerate(targets):
                        packed[target], packed_scales[target] = weights[index], scales[index]
                    self.assertEqual([packed[target] for target in targets], weights)
                    self.assertEqual([packed_scales[target] for target in targets], scales)
                    self.assertEqual(sum(map(len, packed)), sum(map(len, weights)))

    def test_template_retains_every_dot_and_partial_update(self):
        original = TEMPLATE.read_text()
        modified = packed_template(original)
        declaration = next(line for line in modified.splitlines()
                           if line.startswith('ss << "  let logical_lane = '))
        restored = modified.replace(declaration + "\n", "").replace(
            "logical_lane % ", "local_idx % ").replace("logical_lane / ", "local_idx / ").replace(
                "ApplyDensePacked4Template", "ApplyDenseBlock32Template").replace(
                    "((b_global / 4u) * uniforms.K_of_b + k_offset) * 4u + b_global % 4u",
                    "b_global * uniforms.K_of_b + k_offset + b_base_offset").replace(
                        "((b_global / 4u) * uniforms.blocks_per_col + block_idx) * 4u + b_global % 4u",
                        "b_global * uniforms.blocks_per_col + block_idx + b_scale_offset")
        self.assertEqual(restored, original)

    def test_original_kernel_and_grouped_prefill_preserved(self):
        original, header = (PARENT / "matmul_nbits.cc").read_text(), HEADER.read_text()
        modified = transform(original, header)
        signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
        start = original.index(signature)
        end = original.index("\n}\n", start) + 3
        self.assertIn(original[start:end], modified)
        self.assertIn("class GroupedPrefillRows4Program", modified)
        self.assertIn("PersistentPacked4MatMulNBits", modified)
        self.assertIn("weights_constant_ && N_ % 4 == 0", modified)
        self.assertIn("b->IsDataType<uint8_t>()", modified)
        self.assertIn("b->Shape().Size() == N_ * (K_ / 2)", modified)
        self.assertIn("scales->Shape().Size() == N_ * (K_ / 32)", modified)
        self.assertIn("std::call_once(pack_once_", modified)
        self.assertIn('packed_replay_{"DensePacked4", {0}}', modified)
        self.assertIn("{packed_weights_.get(), packed_scales_.get()}", modified)
        self.assertIn("CreateGPUTensor(b->DataType(), b->Shape())", modified)
        self.assertIn("CreateGPUTensor(scales->DataType(), scales->Shape())", modified)
        self.assertNotIn("DenseDecodeWeightsProgram", modified)
        self.assertNotIn("ORT_WEBGPU_DENSE_PREDECODE", modified)
        self.assertIn('Program{"MatMulNBitsPacked4F16"}', modified)

    def test_original_n1_fallback_and_owned_cache_replay_stats(self):
        def row(operator, requests):
            return dict(operator=operator, requests=requests, entries=1,
                        hits=requests - 1, captures=1, rebinds=requests - 2, binding_hits=1,
                        capacity=64, binding_entries=2, binding_capacity=64)
        rows = [row("MatMulNBitsExact", 128) for _ in range(8)] + [row("MatMulNBitsExact", 256)]
        rows += [row("DensePacked4", 128) for _ in range(8)]
        verify_packed_stats(rows)
        rows[8]["entries"] = 2
        with self.assertRaises(AssertionError):
            verify_packed_stats(rows)
        rows[8]["entries"] = 1
        rows[-1]["binding_hits"] = 0
        with self.assertRaises(AssertionError):
            verify_packed_stats(rows)


if __name__ == "__main__":
    unittest.main()
