import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sweep_gdn_pending as sweep


class PendingSweepTests(unittest.TestCase):
    def test_existing_and_combined_stack_drivers(self):
        for kind in ("gdn-view", "prefill-dense"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                calls = []

                def execute(command, stdout, stderr, check):
                    self.assertTrue(check)
                    calls.append(command)
                    prefix = Path(command[command.index("--output-prefix") + 1])
                    key = "result" if kind == "prefill-dense" else "summary"
                    prefix.with_suffix(".qualification.json").write_text(json.dumps({
                        key: dict(median_decode_tps=75, decode_tps=74)}))

                argv = ["sweep", "--kind", kind, "--pending", "64", "256", "64",
                        "--output-prefix", str(root / "test")]
                with patch.object(sweep, "ROOT", root), patch("sys.argv", argv), \
                        patch.object(sweep.subprocess, "run", side_effect=execute), \
                        contextlib.redirect_stdout(io.StringIO()):
                    sweep.main()
                report = json.loads((root / "test.sweep.json").read_text())
                self.assertEqual(report["status"], "completed")
                self.assertEqual([r["pending"] for r in report["rows"]], [64, 256, 64])
                self.assertFalse(report["headline_eligible"])
                self.assertTrue(all(r["median_decode_tps"] == 75 for r in report["rows"]))
                for command in calls:
                    self.assertIn("run_prefill_dense_model.py" if kind == "prefill-dense"
                                  else "run_int4_candidate.py", command[1])
                    if kind == "prefill-dense":
                        self.assertEqual(command[command.index("--chunk") + 1], "128")


if __name__ == "__main__":
    unittest.main()
