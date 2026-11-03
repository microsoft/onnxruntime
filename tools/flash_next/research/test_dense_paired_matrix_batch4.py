from pathlib import Path
import unittest

from build_dense_paired_matrix_batch4 import CLASS, FLAG, PROGRAM, SOURCE, STORAGE_BYTES, transform
from test_hc_projection_inner_unroll import render


class DenseMatrixBatch4Tests(unittest.TestCase):
    def test_actual_renderer_order_and_storage_reuse_barrier(self):
        shader = render(PROGRAM, 1, False, False, helper="", flag=FLAG,
                        class_name=CLASS, constructor_args=())
        self.assertEqual(shader.count("subgroupMatrixMultiplyAccumulate("), 1)
        self.assertEqual(shader.count("workgroupBarrier();"), 6)
        self.assertIn("for (var batch = 0u; batch < 2u; batch++)", shader)
        self.assertIn("let word = batch * 2u + slot / 2u", shader)
        self.assertIn("sums[j] += half_dots[address] + half_dots[address + 512u]", shader)
        self.assertIn("totals[j] += sums[j]", shader)
        self.assertIn("value += partials[local_idx * 32u + stream]", shader)
        self.assertIn("    workgroupBarrier();\n  }\n  if (sg < 4u)", shader)
        self.assertIn("if (correct_pair(raw0))", shader)
        self.assertIn("if (correct_pair(raw1))", shader)
        self.assertEqual(shader.count("{"), shader.count("}"))
        self.assertEqual(STORAGE_BYTES, 46080)

    def test_unique_storage_writers_per_batch(self):
        for batch in range(2):
            half_dots, first, slots = [], [], set()
            for sg in range(16):
                slot, group = sg // 4, sg % 4
                slots.add((batch * 2 + slot // 2, slot % 2))
                for lane in range(32):
                    for j in range(4):
                        item = lane + j * 32
                        stream, col = divmod(item, 16)
                        half_dots.append(slot * 512 + group * 128 + item)
                        first.extend((sg * 256 + stream * 32 + col,
                                      sg * 256 + stream * 32 + col + 16))
            self.assertEqual(sorted(half_dots), list(range(2048)))
            self.assertEqual(sorted(first), list(range(4096)))
            self.assertEqual(slots, {(batch * 2 + word, half) for word in range(2) for half in range(2)})

    def test_owner_reads_preserve_four_word_rounding_order(self):
        partials = []
        for sg in range(4):
            for lane in range(32):
                for j in range(4):
                    item = lane + j * 32
                    words = []
                    for batch in range(2):
                        for word in range(2):
                            address = word * 1024 + sg * 128 + item
                            self.assertLess(address + 512, 2048)
                            words.append(batch * 2 + address // 1024)
                    self.assertEqual(words, [0, 1, 2, 3])
                    stream, col = divmod(item, 16)
                    partials.append(col * 32 + sg * 8 + stream)
        self.assertEqual(sorted(partials), list(range(512)))

    def test_admission_and_unchanged_fallbacks(self):
        source = Path(str(SOURCE).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        candidate = transform(source)
        for text in (f"{FLAG} must be 0 or 1", f"maxComputeWorkgroupStorageSize >= {STORAGE_BYTES}",
                     "maxComputeWorkgroupSizeX >= 512", "maxComputeInvocationsPerWorkgroup >= 512",
                     ".SetWorkgroupSize(512).SetDispatchGroupSize(N_ / 16)",
                     "Type::F16, Type::F32, 16, 16, 16, 32, false"):
            self.assertIn(text, candidate)
        self.assertTrue(candidate.endswith(source[source.index("\nStatus ApplyMatMulNBits("):]))


if __name__ == "__main__":
    unittest.main()
