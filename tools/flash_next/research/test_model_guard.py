import unittest
import json
from pathlib import Path
import sys
import tempfile
import io
import subprocess
from unittest.mock import Mock, patch

from safe_model_run import container_process_ids, gpu_compute_pids, main, stop_container


class GuardTest(unittest.TestCase):
    @patch("safe_model_run.subprocess.check_output")
    def test_explicit_reserved_gpu_is_used_for_memory_and_uuid_queries(self, output):
        output.side_effect = ["GPU-reserved\n", "10000\n"]
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "guard"
            with patch.object(sys, "argv", [
                "guard", "--gpu-index", "1", "--output-prefix", str(prefix),
                "--wait-seconds", "0", "--idle-seconds", "0", "--", "true",
            ]):
                self.assertEqual(main(), 75)
            state = json.loads(prefix.with_suffix(".guard.json").read_text())
        self.assertEqual(state["gpu_index"], 1)
        self.assertEqual(state["gpu_uuid"], "GPU-reserved")
        self.assertEqual(state["gpu_free_mib"], 10000)
        for call in output.call_args_list:
            self.assertIn("--id=1", call.args[0])

    @patch("safe_model_run.subprocess.check_output")
    def test_other_gpu_work_does_not_block_reserved_gpu(self, output):
        output.return_value = "123, GPU-other\n456, GPU-reserved\n456, GPU-reserved\n"
        self.assertEqual(gpu_compute_pids("GPU-reserved"), {456})
        self.assertIn("--id=GPU-reserved", output.call_args.args[0])

    @patch("safe_model_run.subprocess.check_output")
    def test_empty_gpu_is_idle(self, output):
        output.return_value = ""
        self.assertEqual(gpu_compute_pids("GPU-reserved"), set())

    @patch("safe_model_run.subprocess.check_output")
    def test_container_ownership_uses_explicit_pids(self, output):
        output.return_value = "PID\n123\n456\n"
        self.assertEqual(container_process_ids("owned-container"), {123, 456})
        self.assertEqual(output.call_args.args[0][-4:], ["top", "owned-container", "-eo", "pid"])

    @patch("safe_model_run.time.sleep")
    @patch("safe_model_run.subprocess.run")
    def test_cleanup_retries_transient_pam_failure_for_only_owned_container(self, run, sleep):
        run.side_effect = [
            subprocess.CompletedProcess([], 1, "", "sudo: PAM account management error: Unknown error -1\n"),
            subprocess.CompletedProcess([], 0, "owned-container\n", ""),
        ]
        log = io.StringIO()
        self.assertTrue(stop_container("owned-container", log))
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(call.args[0][-4:] == ["stop", "--time", "0", "owned-container"]
                            for call in run.call_args_list))
        sleep.assert_called_once_with(1)
        self.assertIn("PAM account management error", log.getvalue())

    @patch("safe_model_run.subprocess.run")
    def test_cleanup_surfaces_nontransient_error(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, "", "daemon unavailable")
        with self.assertRaises(subprocess.CalledProcessError):
            stop_container("owned-container", io.StringIO())
        run.assert_called_once()

    @patch("safe_model_run.time.sleep")
    @patch("safe_model_run.subprocess.run")
    def test_cleanup_retry_count_is_bounded(self, run, sleep):
        run.return_value = subprocess.CompletedProcess([], 1, "", "sudo: PAM account management error: unknown")
        with self.assertRaises(subprocess.CalledProcessError):
            stop_container("owned-container", io.StringIO())
        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    @patch("safe_model_run.subprocess.run")
    def test_already_exited_container_is_not_reported_as_stopped(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, "", "No such container: owned-container")
        self.assertFalse(stop_container("owned-container", io.StringIO()))

    def test_timeout_and_failed_cleanup_are_persisted(self):
        def docker_result(command, **kwargs):
            if command[0] == "bash":
                return subprocess.CompletedProcess(command, 0, "owned-container\n", "")
            if "inspect" in command:
                return subprocess.CompletedProcess(command, 0, "123\n", "")
            return subprocess.CompletedProcess(command, 1, "", "sudo: PAM account management error: unknown")

        process = Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "guard"
            with (
                patch.object(sys, "argv", [
                    "guard", "--output-prefix", str(prefix), "--idle-seconds", "0",
                    "--run-seconds", "0", "--", "true",
                ]),
                patch("safe_model_run.subprocess.check_output",
                      side_effect=["GPU-reserved\n", "140000\n", "snapshot"]),
                patch("safe_model_run.gpu_compute_pids", return_value=set()),
                patch("safe_model_run.subprocess.run", side_effect=docker_result),
                patch("safe_model_run.subprocess.Popen", return_value=process),
                patch("safe_model_run.time.monotonic", side_effect=range(100)),
                patch("safe_model_run.time.sleep"),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    main()
            state = json.loads(prefix.with_suffix(".guard.json").read_text())
        self.assertEqual(state["status"], "run_timed_out")
        self.assertEqual(state["cleanup_status"], "failed_process_still_running")


if __name__ == "__main__":
    unittest.main()
