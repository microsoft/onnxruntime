from pathlib import Path
import unittest

from prepare_literal_snapshot_native import transform


class FixedStartupNativeTests(unittest.TestCase):
    def test_fixed_flags_and_independent_shapes_bytes_and_eof(self):
        source = Path(
            "/home/sayanshaw/repos/qwen38-int4-qmoe-unused-scratch/native-v2/native_compare.cpp")
        rendered = transform(source.read_text())
        self.assertNotIn("setenv(", rendered)
        self.assertNotIn("set_flag(", rendered)
        self.assertIn("for (int file = 4;", rendered)
        self.assertIn("O_EXCL", rendered)
        self.assertIn("expected_shape != shape", rendered)
        self.assertIn("expected_count != count", rendered)
        self.assertIn("std::memcmp(&expected, &value, sizeof(value)) != 0", rendered)
        self.assertIn("std::fgetc(golden) != EOF", rendered)
        self.assertIn("std::fclose(golden) != 0", rendered)
        self.assertIn("PASS total_finite_elements=", rendered)
        self.assertIn("PASS total_exact_elements=", rendered)


if __name__ == "__main__":
    unittest.main()
