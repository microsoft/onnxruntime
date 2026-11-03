import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_cpu_decode_sample import instrument_requests
from run_cpu_decode_sample import owned_model_pid


class CpuDecodeSampleTests(unittest.TestCase):
    def test_only_first_measured_request_is_sampled(self):
        seen = []

        def original(value, **kwargs):
            seen.append(kwargs)
            for index in range(256):
                callback = kwargs.get("on_decode_step")
                if callback is not None:
                    callback(index, float(index))
            return value

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "markers"
            wrapped = instrument_requests(original, prefix)
            self.assertEqual([wrapped(i) for i in range(4)], list(range(4)))
            self.assertEqual([bool(x) for x in seen], [False, True, False, False])
            for phase, index in (("start", 32), ("stop", 224)):
                raw = json.loads(Path(str(prefix) + f".{phase}.json").read_text())
                self.assertEqual((raw["phase"], raw["token_index"], raw["request_index"]), (phase, index, 1))

    def test_existing_callback_is_not_silently_overridden(self):
        wrapped = instrument_requests(lambda **kwargs: None, Path("/unused"))
        wrapped()
        with self.assertRaises(ValueError):
            wrapped(on_decode_step=lambda index, timestamp: None)

    def test_attachment_requires_container_pid_user_and_exact_script(self):
        status = f"Uid:\t{os.getuid()} 0 0 0\nNSpid:\t123 1\n"
        with patch("run_cpu_decode_sample.container_process_ids", return_value={123}), \
                patch.object(Path, "read_text", return_value=status), \
                patch.object(Path, "read_bytes", return_value=b"python3\0/evidence/benchmark_cpu_decode_sample.py\0"):
            self.assertEqual(owned_model_pid("owned-container", {"container_pid": 1}), 123)
            with self.assertRaises(RuntimeError):
                owned_model_pid("owned-container", {"container_pid": 2})

    def test_foreign_user_cannot_be_sampled(self):
        with patch("run_cpu_decode_sample.container_process_ids", return_value={123}), \
                patch.object(Path, "read_text", return_value=f"Uid:\t{os.getuid() + 1} 0 0 0\nNSpid:\t123 1\n"):
            with self.assertRaises(RuntimeError):
                owned_model_pid("owned-container", {"container_pid": 1})


if __name__ == "__main__":
    unittest.main()
