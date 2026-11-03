from pathlib import Path
import unittest

from build_qmoe_fast_flags import HELPER, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/build-v2/qmoe.cc")


class FastFlagTests(unittest.TestCase):
    def test_only_parser_calls_and_per_compute_setup_change(self):
        source = SOURCE.read_text()
        actual = transform(source)
        prefix, body = actual.split("Status QMoE::ComputeInternal(", 1)
        self.assertIn(HELPER, prefix)
        self.assertEqual(body.count('const int fast_flags ='), 2)
        self.assertEqual(body.count("read_flag("), 24)
        self.assertIn("std::getenv(name)", HELPER)
        self.assertNotIn("static ", HELPER)
        self.assertIn("return ParseEnvironmentVariableWithDefault<int>(name, default_value);", HELPER)
        self.assertIn("value[1] == '\\0'", HELPER)
        original_body = source[source.index("Status QMoE::ComputeInternal("):]
        self.assertEqual(original_body.count("key.push_back("), body.count("key.push_back("))
        for line in original_body.splitlines():
            if "ParseEnvironmentVariableWithDefault<int>" not in line:
                self.assertIn(line, actual)


if __name__ == "__main__":
    unittest.main()
