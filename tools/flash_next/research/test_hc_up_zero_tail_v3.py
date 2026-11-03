from pathlib import Path
import unittest

from prepare_hc_up_zero_tail_v3 import controller_source, fixture_source


ROOT = Path(__file__).parent


class HCUpProtectedHardlinkTests(unittest.TestCase):
    def test_only_expected_permission_error_gets_explicit_repository_ln_fallback(self):
        actual = fixture_source((ROOT / "prepare_hc_up_zero_tail_actual_v2.py").read_text())
        self.assertIn("except PermissionError:", actual)
        self.assertIn('subprocess.run(["sudo", "-n", "ln", str(source), str(alias)], check=True)', actual)
        self.assertNotIn("except Exception", actual)
        self.assertNotIn("copyfile", actual)
        self.assertNotIn("symlink_to", actual)
        self.assertIn("alias.samefile(source)", actual)
        self.assertIn("range_sha(source, offset, length)", actual)

    def test_controller_diff_is_only_fresh_packaging_names(self):
        original = (ROOT / "run_hc_up_zero_tail_v2.py").read_text()
        actual = controller_source(original)
        restored = actual.replace("actual-screen-v3", "actual-screen-v2")
        restored = restored.replace('"hc-up-zero-tail-v3"', '"hc-up-zero-tail-v2"')
        restored = restored.replace("hc-up-zero-tail-actual-v3-generation.json",
                                    "hc-up-zero-tail-actual-v2-generation.json")
        restored = restored.replace("from prepare_hc_up_zero_tail_actual_v3 import",
                                    "from prepare_hc_up_zero_tail_actual_v2 import")
        self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
