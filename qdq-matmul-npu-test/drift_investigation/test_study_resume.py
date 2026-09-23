import tempfile
import unittest
from pathlib import Path

from drift_investigation.study_resume import completed, next_attempt_dir


class StudyResumeTest(unittest.TestCase):
    def test_retries_keep_failed_and_interrupted_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = next_attempt_dir(root, "case")
            (first / "partial.log").write_text("failed", encoding="utf-8")
            retry = next_attempt_dir(root, "case")
            self.assertEqual(retry.name, "case.retry-1")
            self.assertEqual((first / "partial.log").read_text(encoding="utf-8"), "failed")
            self.assertEqual(next_attempt_dir(root, "case").name, "case.retry-2")
        self.assertFalse(completed(None))
        self.assertFalse(completed({"error": "compilation failed"}))
        self.assertTrue(completed({"cpu_shift_mae": 1.0}))


if __name__ == "__main__":
    unittest.main()
