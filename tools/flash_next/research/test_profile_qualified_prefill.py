import copy
from pathlib import Path
import unittest

from profile_qualified_prefill import profile_command


class QualifiedPrefillProfileTests(unittest.TestCase):
    def test_only_output_count_repetitions_and_profiling_change(self):
        original = ["bash", "-lc", "export A=1; runner --tokens 256 --repetitions 3 --chunk 1024 "
                    "--webgpu-validation-mode full --output baseline.json STATS=baseline-stats STATIC=baseline-static"]
        preserved = copy.deepcopy(original)
        actual = profile_command(original, Path("/evidence/profile-test"), "baseline")
        self.assertEqual(original, preserved)
        self.assertEqual(actual, ["bash", "-lc",
            "export A=1; runner --tokens 2 --repetitions 1 --chunk 1024 --webgpu-validation-mode full "
            "--output profile-test.json STATS=profile-test-stats STATIC=profile-test-static "
            "--profile-prefix /evidence/webgpu-perf-oct20/profile-test-profile"])

    def test_missing_ambiguous_or_already_profiled_commands_are_rejected(self):
        for text in ("runner", "runner --tokens 256 --tokens 256 --repetitions 3",
                     "runner --tokens 256 --repetitions 3 --profile-prefix old",
                     "runner --tokens 256 --repetitions 3 --profile-decode-after 35"):
            with self.assertRaises(AssertionError):
                profile_command(["bash", "-lc", text], Path("/evidence/unused"), "baseline")

    def test_original_evidence_cannot_be_reused(self):
        text = "runner --tokens 256 --repetitions 3 --output baseline.json STATS=baseline-stats STATIC=baseline-static"
        with self.assertRaises(AssertionError):
            profile_command(["bash", "-lc", text], Path("/evidence/baseline"), "baseline")
        with self.assertRaises(AssertionError):
            profile_command(["bash", "-lc", text.replace(" STATIC=baseline-static", "")],
                            Path("/evidence/new"), "baseline")


if __name__ == "__main__":
    unittest.main()
