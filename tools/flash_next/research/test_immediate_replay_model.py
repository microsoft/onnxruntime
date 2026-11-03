import json
import unittest

from run_immediate_replay_model import DIRECTORY, FLAG, ROOT, TEMPLATE, model_command, prefix_for


class ImmediateReplayModelTests(unittest.TestCase):
    def test_standard_keeps_sampling_submission_limit_and_default_worker_policy(self):
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for mode in ("control", "immediate"):
            command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                    prefix_for(mode, 2048, True, 0), 2048, True, mode, "standard")
            self.assertIn(f"export {FLAG}={int(mode == 'immediate')}; ", command[-1])
            self.assertIn("--tokens 256 --repetitions 3", command[-1])
            self.assertIn("--webgpu-max-pending-dispatches 64", command[-1])
            self.assertNotIn("--inherit-worker-cpu-affinity", command[-1])
            self.assertNotIn("ORT_GENAI_WEBGPU_ALLOCATOR_SINGLE_THREAD", command[-1])

    def test_coverage_does_not_disable_replay_with_gpu_profiling(self):
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                prefix_for("immediate", 2048, True, 0, "coverage"),
                                2048, True, "immediate", "coverage")
        self.assertIn("--tokens 36 --repetitions 1", command[-1])
        self.assertNotIn("--profile", command[-1])
        self.assertIn("ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=", command[-1])


if __name__ == "__main__":
    unittest.main()
