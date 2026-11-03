import unittest
from unittest.mock import patch

from cpu_local_windows import CpuLocalWindows
import test_qsa_rank_decode_windows as fixtures
import qsa_rank_decode_windows as windows


class CpuLocalTests(unittest.TestCase):
    def setUp(self):
        self.allowed = set(range(96))
        self.moves = []

    def get(self, pid):
        self.assertEqual(pid, 0)
        return self.allowed.copy()

    def put(self, pid, cpus):
        self.assertEqual(pid, 0)
        self.allowed = set(cpus)
        self.moves.append(sorted(cpus))

    def test_only_main_thread_moves_and_control_restores(self):
        with patch("os.sched_getaffinity", self.get), patch("os.sched_setaffinity", self.put):
            mode = CpuLocalWindows()
            mode.switch(1)
            self.assertEqual(self.allowed, set(range(48, 96)))
            mode.switch(0)
            self.assertEqual(self.allowed, set(range(96)))
            mode.switch(1)
            mode.restore()
            self.assertEqual(self.allowed, set(range(96)))

    def test_no_silent_fallback_for_invalid_or_ineffective_affinity(self):
        for allowed in (set(range(48)), set(range(48, 96))):
            with patch("os.sched_getaffinity", return_value=allowed):
                with self.assertRaises(RuntimeError):
                    CpuLocalWindows()
        with patch("os.sched_getaffinity", self.get), patch("os.sched_setaffinity"):
            mode = CpuLocalWindows()
            with self.assertRaises(RuntimeError):
                mode.switch(1)

    def test_complementary_windows_restore_affinity_and_flag_on_success_and_interruption(self):
        for interrupt in (False, True):
            fixture = fixtures.RankWindowTests()
            with patch("os.sched_getaffinity", self.get), patch("os.sched_setaffinity", self.put):
                if interrupt:
                    with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
                        fixture.invoke(experiment_flag=windows.CPU_LOCAL_FLAG, interrupt=True)
                else:
                    result, _ = fixture.invoke(experiment_flag=windows.CPU_LOCAL_FLAG)
                    self.assertEqual(result["window_protocol"]["cpu_affinity"]["candidate_cpus"],
                                     list(range(48, 96)))
                    self.assertEqual(result["window_summary"]["32768"]["cpu_local1_median_window_tps"], 2.0)
                self.assertEqual(self.allowed, set(range(96)))


if __name__ == "__main__":
    unittest.main()
