import unittest

from run_webgpu_cpu_perf import UUID, clone_command, owned_profile_pid


class CpuProfileOwnershipTests(unittest.TestCase):
    def test_command_clone_rehomes_both_statistics_and_output(self):
        body = " ".join(f"/evidence/webgpu-perf-oct20/old{suffix}"
                        for suffix in (".json", "-stats.jsonl", "-static-stats.jsonl"))
        original = ["bash", "-lc", body]
        result = clone_command(original, "old", "new")
        self.assertEqual(original[-1], body)
        self.assertEqual(result[-1], body.replace("old", "new"))
        with self.assertRaises(AssertionError):
            clone_command(["bash", "-lc", body + " /evidence/webgpu-perf-oct20/old.json"], "old", "new")

    def setUp(self):
        self.state = dict(status="running", gpu_index=7, gpu_uuid=UUID, own_compute_pids=[123])

    def test_only_the_owned_container_compute_pid_is_accepted(self):
        self.assertEqual(owned_profile_pid(self.state, {122, 123}, {123}), 123)

    def test_rejects_foreign_missing_and_ambiguous_pids(self):
        for container, gpu in (({122}, {123}), ({123}, {124}), ({123, 124}, {123, 124})):
            with self.assertRaises(AssertionError):
                owned_profile_pid(self.state, container, gpu)

    def test_rejects_completed_guard_and_wrong_physical_gpu(self):
        for key, value in (("status", "completed"), ("gpu_index", 4), ("gpu_uuid", "foreign"),
                           ("own_compute_pids", [])):
            state = dict(self.state)
            state[key] = value
            with self.assertRaises(AssertionError):
                owned_profile_pid(state, {123}, {123})


if __name__ == "__main__":
    unittest.main()
