import unittest

from prepare_qmoe_prefill_rows4_double_buffer_native import comparator_source, SOURCE


class IndependentPrefillTests(unittest.TestCase):
    def test_original_schedule_and_independent_codec(self):
        source = SOURCE.read_text()
        candidate = comparator_source()
        start = "      for (int64_t rows :"
        end = '      flag("2");'
        original = source[source.index(start):source.index(end)]
        changed = candidate[candidate.index(start):candidate.index("      if (!capture")]
        transfer_start = changed.index("          TransferOriginalOutput(")
        transfer_end = changed.index("          for (size_t i =", transfer_start)
        changed = changed[:transfer_start] + changed[transfer_end:]
        self.assertEqual(original.replace(
            'throw std::runtime_error(std::string(argv[file]) + " finite/bitwise mismatch rows="',
            'throw NumericMismatch(std::string(argv[file]) + " finite/bitwise mismatch rows="'), changed)
        self.assertIn("for (int file = 4; file < argc; ++file)", candidate)
        self.assertIn("file % 2", candidate)
        self.assertIn("return 2;", candidate)
        self.assertIn("Trailing original reference bytes", candidate)
        self.assertNotIn("ORT_WEBGPU_QMOE_PREFILL_ROWS4_DOUBLE_BUFFER", candidate)


if __name__ == "__main__":
    unittest.main()
