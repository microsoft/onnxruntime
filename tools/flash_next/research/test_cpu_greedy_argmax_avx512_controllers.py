import json
import unittest

from run_cpu_greedy_argmax_avx512_standard import BENCHMARK, BUILD, FLAG, PROVIDER, ROOT, TEMPLATE, command_for, prefix_for


class AVX512ControllerTests(unittest.TestCase):
    def test_identical_runtime_gpu_and_cpu_policy(self):
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for context in (2048, 8192, 32768):
            for full in (False, True):
                with self.subTest(context=context, full=full):
                    commands = [command_for(template, prefix_for(mode, context, full, 0),
                                            context, full, mode) for mode in ("control", "avx512")]
                    for flag, command in enumerate(commands):
                        tail = command[-1]
                        self.assertIn(f"export {FLAG}={flag}; ", tail)
                        self.assertIn(str(BUILD).replace("/home/sayanshaw/repos", "/workspace"), tail)
                        self.assertIn(str(PROVIDER).replace("/home/sayanshaw/repos", "/workspace"), tail)
                        self.assertIn(BENCHMARK.name, tail)
                        self.assertIn("--inherit-worker-cpu-affinity", tail)
                        self.assertIn("--tokens 256", tail)
                        self.assertIn("--repetitions 3", tail)
                        self.assertEqual(command[:3], ["taskset", "-c", "48-55"])
                    normalized = []
                    for flag, mode in enumerate(("control", "avx512")):
                        prefix = prefix_for(mode, context, full, 0)
                        normalized.append(commands[flag][-1].replace(f"export {FLAG}={flag}; ", "")
                                          .replace(prefix.name, "MATCHED_PREFIX"))
                    self.assertEqual(normalized[0], normalized[1])

    def test_invalid_mode_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown CPU argmax mode"):
            command_for([], ROOT / "unused", 2048, True, "invalid")


if __name__ == "__main__":
    unittest.main()
