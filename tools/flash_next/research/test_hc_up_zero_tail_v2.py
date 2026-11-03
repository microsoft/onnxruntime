from pathlib import Path
import unittest

from prepare_hc_up_zero_tail_v2 import controller_source, fixture_source


ROOT = Path(__file__).parent


class HCUpHardlinkPackagingTests(unittest.TestCase):
    def test_only_packaging_and_fresh_names_change_the_fixture_generator(self):
        actual = fixture_source((ROOT / "prepare_hc_up_zero_tail_actual.py").read_text())
        self.assertIn("alias.hardlink_to(source)", actual)
        self.assertNotIn("symlink_to", actual)
        self.assertNotIn("copyfile", actual)
        self.assertIn("alias.samefile(source)", actual)
        self.assertIn("onnx.checker.check_model(str(path))", actual)
        self.assertIn("range_sha(source, offset, length)", actual)
        self.assertIn("actual-fixtures-v2", actual)
        self.assertIn("no_model_weight_copies", actual)

    def test_shader_build_program_and_exact_scope_are_not_changed(self):
        original = (ROOT / "run_hc_up_zero_tail_v1.py").read_text()
        actual = controller_source(original)
        restored = actual.replace("actual-screen-v2", "actual-screen-v1")
        restored = restored.replace('"hc-up-zero-tail-v2"', '"hc-up-zero-tail-v1"')
        restored = restored.replace("hc-up-zero-tail-actual-v2-generation.json",
                                    "hc-up-zero-tail-actual-v1-generation.json")
        restored = restored.replace("from prepare_hc_up_zero_tail_actual_v2 import",
                                    "from prepare_hc_up_zero_tail_actual import")
        self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
