from pathlib import Path
import unittest

from build_dense_paired_matrix_batch8 import CLASS, FLAG, PROGRAM, SOURCE, STORAGE_BYTES, transform
from test_hc_projection_inner_unroll import render


class DenseMatrixBatch8Tests(unittest.TestCase):
    def test_actual_cpp_renderer_and_rounding(self):
        shader = render(PROGRAM, 1, False, False, helper="", flag=FLAG,
                        class_name=CLASS, constructor_args=())
        self.assertEqual(shader.count("subgroupMatrixMultiplyAccumulate("), 1)
        self.assertEqual(shader.count("workgroupBarrier();"), 5)
        self.assertNotIn("for (var half =", shader)
        self.assertNotIn("for (var word =", shader)
        self.assertIn("sum += half_dots[address] + half_dots[address + 512u]", shader)
        self.assertIn("totals[j] += sum", shader)
        self.assertIn("value += partials[local_idx * 32u + stream]", shader)
        self.assertIn("if (correct_pair(raw0))", shader)
        self.assertIn("if (correct_pair(raw1))", shader)
        self.assertEqual(shader.count("{"), shader.count("}"))
        self.assertEqual(STORAGE_BYTES, 91136)

    def test_unique_half_dot_and_matrix_storage_ownership(self):
        half_dots = []
        first = []
        for sg in range(32):
            slot, group = sg // 4, sg % 4
            self.assertEqual((slot // 2, slot % 2), (sg // 8, (sg // 4) % 2))
            for lane in range(32):
                for j in range(4):
                    item = lane + j * 32
                    stream, col = divmod(item, 16)
                    half_dots.append(slot * 512 + group * 128 + item)
                    first.extend((sg * 256 + stream * 32 + col,
                                  sg * 256 + stream * 32 + col + 16))
        self.assertEqual(sorted(half_dots), list(range(4096)))
        self.assertEqual(sorted(first), list(range(8192)))

    def test_owner_reduction_reads_original_word_half_order(self):
        partials = []
        for sg in range(4):
            for lane in range(32):
                for j in range(4):
                    item = lane + j * 32
                    addresses = [(word * 1024 + sg * 128 + item,
                                  word * 1024 + sg * 128 + item + 512) for word in range(4)]
                    self.assertEqual([(address // 512) // 2 for pair in addresses for address in pair],
                                     [0, 0, 1, 1, 2, 2, 3, 3])
                    self.assertLess(max(address for pair in addresses for address in pair), 4096)
                    stream, col = divmod(item, 16)
                    partials.append(col * 32 + sg * 8 + stream)
        self.assertEqual(sorted(partials), list(range(512)))

    def test_admission_and_original_fallbacks(self):
        source = Path(str(SOURCE).replace("/workspace/", "/home/sayanshaw/repos/")).read_text()
        candidate = transform(source)
        for text in (f"{FLAG} must be 0 or 1", f"maxComputeWorkgroupStorageSize >= {STORAGE_BYTES}",
                     "maxComputeWorkgroupSizeX >= 1024", "maxComputeInvocationsPerWorkgroup >= 1024",
                     ".SetWorkgroupSize(1024).SetDispatchGroupSize(N_ / 16)",
                     "Type::F16, Type::F32, 16, 16, 16, 32, false"):
            self.assertIn(text, candidate)
        self.assertTrue(candidate.endswith(source[source.index("\nStatus ApplyMatMulNBits("):]))


if __name__ == "__main__":
    unittest.main()
