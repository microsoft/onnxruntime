import unittest

from qmoe_prompt_runner import first_free_gpu_index, parse_free_gpu_indices


class TestParseFreeGpuIndices(unittest.TestCase):
    def test_returns_sorted_free_gpu_indices(self):
        output = """\
2, 0
0, 1024
1, 0
"""
        self.assertEqual(parse_free_gpu_indices(output), [1, 2])

    def test_returns_empty_list_when_all_gpus_are_used(self):
        self.assertEqual(parse_free_gpu_indices("0, 1\n1, 2048\n"), [])

    def test_rejects_malformed_output(self):
        with self.assertRaisesRegex(ValueError, "Unexpected nvidia-smi output"):
            parse_free_gpu_indices("0\n")


class TestFirstFreeGpuIndex(unittest.TestCase):
    def test_selects_lowest_index(self):
        self.assertEqual(first_free_gpu_index([5, 2, 7]), 2)

    def test_raises_when_no_gpu_is_free(self):
        with self.assertRaisesRegex(RuntimeError, "No free GPU found"):
            first_free_gpu_index([])


if __name__ == "__main__":
    unittest.main()
