import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_avx512_hc_parallel16_progression as runner


class ProgressionTests(unittest.TestCase):
    def run_progression(self, statuses):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def execute(script, *arguments):
                calls.append((script, arguments))
                context = int(arguments[1])
                repeat = "--repeat" in arguments
                path = root / f"test-context{context}-{'repeat-' if repeat else ''}screen-stamp.qualification.json"
                path.write_text(json.dumps(dict(status=statuses[len(calls) - 1],
                                                source_sha256={}, rows=[])))

            with patch.object(runner, "ROOT", root), patch.object(runner, "execute", execute):
                result = runner.progression("test.py", "test", "stamp", "initial", "repeat")
        return result, calls

    def test_initial_failure_stops_without_repeat_or_escalation(self):
        result, calls = self.run_progression(["no_supported_decode_gain"])
        self.assertFalse(result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], ("--context", "2048"))

    def test_repeat_failure_stops_before_longer_context(self):
        result, calls = self.run_progression(["initial", "no_supported_decode_gain"])
        self.assertFalse(result)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1], ("--context", "2048", "--repeat"))

    def test_complete_initial_repeat_order(self):
        result, calls = self.run_progression(["initial", "repeat"] * 3)
        self.assertTrue(result)
        self.assertEqual([arguments for _, arguments in calls],
                         [("--context", "2048"), ("--context", "2048", "--repeat"),
                          ("--context", "8192"), ("--context", "8192", "--repeat"),
                          ("--context", "32768"), ("--context", "32768", "--repeat")])


if __name__ == "__main__":
    unittest.main()
