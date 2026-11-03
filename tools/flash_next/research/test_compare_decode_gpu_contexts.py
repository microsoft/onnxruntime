import copy
import unittest

from compare_decode_gpu_contexts_v1 import WORKGROUPS, summarize


class ContextComparisonTests(unittest.TestCase):
    def profiles(self):
        return [[dict(cat="Kernel", pid=-1, tid=-1, ph="X", name="node&" + name, dur=2,
                      args={"cache_key": f"{name}[hint]:{threads},1,1:type"})
                 for name, threads in WORKGROUPS.items() for _ in range(48)] for _ in range(4)]

    def test_exact_token_denominator_and_program_grouping(self):
        result = summarize(self.profiles())
        self.assertEqual(result["gpu_events_per_token"], 144)
        self.assertEqual(result["summed_instrumented_gpu_us_per_token"], 288)
        self.assertEqual(result["program_gpu_us_per_token"], dict.fromkeys(WORKGROUPS, 96))
        self.assertEqual(result["program_dispatches_per_token"], dict.fromkeys(WORKGROUPS, 48))

    def test_missing_tokens_layers_and_wrong_workgroups_reject(self):
        profiles = self.profiles()
        with self.assertRaises(RuntimeError):
            summarize(profiles[:3])
        broken = copy.deepcopy(profiles)
        broken[0].pop()
        with self.assertRaises(RuntimeError):
            summarize(broken)
        profiles[0][0]["args"]["cache_key"] = "program[hint]:32,1,1:type"
        with self.assertRaises(RuntimeError):
            summarize(profiles)


if __name__ == "__main__":
    unittest.main()
