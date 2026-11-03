import json
import unittest
from pathlib import Path
from unittest import mock

import observe_model_run as observer


class ObserverTests(unittest.TestCase):
    def test_transient_incomplete_json_is_retried(self):
        with mock.patch.object(Path, "read_text", side_effect=["", '{"status":"running"}']), \
                mock.patch.object(observer.time, "sleep"):
            self.assertEqual(observer.read_json(Path("unused")), {"status": "running"})

    def test_persistently_invalid_json_fails(self):
        with mock.patch.object(Path, "read_text", return_value="{"), \
                mock.patch.object(observer.time, "sleep"):
            with self.assertRaises(json.JSONDecodeError):
                observer.read_json(Path("unused"))

    def test_absent_startup_file_is_distinct_from_invalid_json(self):
        with mock.patch.object(Path, "read_text", side_effect=FileNotFoundError):
            self.assertIsNone(observer.read_json(Path("unused")))

    def test_proc_stat_fields_with_parentheses_in_command(self):
        status = "Name:\tworker\nVmRSS:\t100 kB\nVmSwap:\t0 kB\nThreads:\t2\n"
        stat = "42 (command with ) parentheses) R " + " ".join(str(i) for i in range(1, 16))
        with mock.patch.object(Path, "read_text", side_effect=[status, stat]):
            sample = observer.process_sample(42)
        self.assertEqual(sample["minor_faults"], 7)
        self.assertEqual(sample["major_faults"], 9)
        self.assertEqual(sample["user_ticks"], 11)
        self.assertEqual(sample["system_ticks"], 12)
        self.assertEqual(len(sample["status"]), 3)


if __name__ == "__main__":
    unittest.main()
