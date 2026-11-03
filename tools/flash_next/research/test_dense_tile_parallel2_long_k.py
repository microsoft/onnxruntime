from pathlib import Path
import unittest

from dense_tile_parallel2_transform import PROGRAM
from dense_tile_parallel2_long_k_transform import FLAG, transform
from prepare_dense_tile_parallel2_long_k import controller_source


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8-double-buffer/build-v1/matmul_nbits.cc")


class DenseTwoWorkerLongKTests(unittest.TestCase):
    def test_only_admission_changes_the_already_measured_two_worker_shader(self):
        actual = transform(SOURCE.read_text())
        self.assertIn(PROGRAM, actual)
        self.assertIn("N_ == 2560 && K_ == 6144 &&", actual)
        self.assertIn(f'"{FLAG}", 0', actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("maxComputeWorkgroupSizeX >= 256", actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 11264", actual)
        self.assertNotIn("(K_ == 2560 || K_ == 6144)", actual)
        self.assertIn("use_tile_parallel2 ? 8 : 7", actual)

    def test_only_long_k_profiles_and_gates_are_selected_and_stale_status_is_corrected_fresh(self):
        actual = controller_source((ROOT / "run_dense_tile_parallel2_v1.py").read_text())
        self.assertIn("bool(selected and (k, n) == (6144, 2560))", actual)
        self.assertIn('"selected": (k, n) == (6144, 2560)', actual)
        self.assertIn("dense_tile_parallel2_long_k_actual_operator_screen_completed", actual)
        self.assertNotIn("dense_tile_parallel3_actual_operator_screen_completed", actual)
        self.assertIn("positive >= 120", actual)
        self.assertIn("ratio > 1.01", actual)


if __name__ == "__main__":
    unittest.main()
