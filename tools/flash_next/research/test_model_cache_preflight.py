import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import model_cache_preflight as cache


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        for name in cache.DATA_FILES:
            (self.model / name).write_bytes(bytes(4096))
        self.hot = [{"file": str(self.model / name), "res": 4096, "size": 4096}
                    for name in cache.DATA_FILES]
        self.cold = [dict(row, res=0) for row in self.hot]
        self.prefix = self.root / "candidate"

    def test_snapshot_uses_privileged_exact_file_probe(self):
        with patch.object(cache.subprocess, "check_output",
                          return_value=json.dumps({"fincore": self.hot})) as probe:
            self.assertEqual(cache.snapshot_model_cache(self.model), self.hot)
        self.assertEqual(probe.call_args.args[0][:3], ["sudo", "-n", "fincore"])
        self.assertEqual(probe.call_args.args[0][-2:], [row["file"] for row in self.hot])

    def test_snapshot_rejects_bad_files_counts_and_sizes(self):
        invalid = [self.hot[:1], [self.hot[0], self.hot[0]],
                   [dict(self.hot[0], file="other"), self.hot[1]],
                   [dict(self.hot[0], size=8192), self.hot[1]],
                   [dict(self.hot[0], res=-1), self.hot[1]],
                   [dict(self.hot[0], res=8192), self.hot[1]],
                   [dict(self.hot[0], res="4096"), self.hot[1]]]
        for rows in invalid:
            with self.subTest(rows=rows), patch.object(cache.subprocess, "check_output",
                    return_value=json.dumps({"fincore": rows})), self.assertRaises(ValueError):
                cache.snapshot_model_cache(self.model)

    def test_probe_errors_propagate(self):
        with patch.object(cache.subprocess, "check_output",
                side_effect=subprocess.CalledProcessError(1, ["fincore"])), \
                self.assertRaises(subprocess.CalledProcessError):
            cache.snapshot_model_cache(self.model)

    def test_hot_data_skips_warming_and_records_evidence(self):
        with patch.object(cache, "snapshot_model_cache", return_value=self.hot), \
                patch.object(cache.subprocess, "run") as warm:
            report = cache.prepare_model_cache(self.model, self.prefix)
        warm.assert_not_called()
        self.assertEqual(report["status"], "ready")
        self.assertFalse(report["warmed"])
        self.assertTrue(report["warming_excluded_from_timing"])
        self.assertEqual(json.loads(Path(report["evidence_path"]).read_text()), report)

    def test_cold_data_is_warmed_then_rechecked(self):
        with patch.object(cache, "snapshot_model_cache", side_effect=[self.cold, self.hot]), \
                patch.object(cache.subprocess, "run") as warm:
            report = cache.prepare_model_cache(self.model, self.prefix)
        self.assertTrue(report["warmed"])
        self.assertEqual(report["status"], "ready")
        self.assertEqual(warm.call_args.args[0][-2:], ["--model", str(self.model)])
        self.assertEqual(warm.call_args.kwargs["timeout"], 600)
        self.assertTrue(warm.call_args.kwargs["check"])

    def test_still_cold_is_explicit_failure(self):
        with patch.object(cache, "snapshot_model_cache", return_value=self.cold), \
                patch.object(cache.subprocess, "run"), self.assertRaisesRegex(RuntimeError, "not fully resident"):
            cache.prepare_model_cache(self.model, self.prefix)
        report = json.loads((self.root / "candidate-cache.json").read_text())
        self.assertEqual(report["status"], "failed_residency")

    def test_warming_failure_is_not_hidden(self):
        with patch.object(cache, "snapshot_model_cache", return_value=self.cold), \
                patch.object(cache.subprocess, "run",
                             side_effect=subprocess.CalledProcessError(1, ["warm"])), \
                self.assertRaises(subprocess.CalledProcessError):
            cache.prepare_model_cache(self.model, self.prefix)
        report = json.loads((self.root / "candidate-cache.json").read_text())
        self.assertEqual(report["status"], "warming")
        self.assertTrue(Path(report["warming_log"]).is_file())


if __name__ == "__main__":
    unittest.main()
