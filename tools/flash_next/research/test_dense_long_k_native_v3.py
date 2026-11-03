import unittest

from prepare_dense_long_k_native_v3 import ROOT, controller_source, native_source


class SplitReplayNativeTests(unittest.TestCase):
    def test_profile_proof_and_real_replay_have_separate_explicit_contracts(self):
        controller = controller_source((ROOT / "run_dense_long_k_native_v2.py").read_text())
        scope = {"__name__": "native_split_test"}
        exec(compile(controller, "split-profile-replay-controller", "exec"), scope)
        verify = scope["verify_replay"]
        verify([], 0, 128, 1)
        verify([], 1, 128, 0)
        rows = [{"operator": "MatMulNBitsExact", "requests": 128, "hits": 127,
                 "binding_hits": 127, "captures": 1} for _ in range(9)]
        verify(rows, 0, 128, 0)
        for bad, fp32, profiling in (([], 0, 0), (rows * 3, 0, 0), (rows, 0, 1), (rows, 1, 0)):
            with self.assertRaises(RuntimeError):
                verify(bad, fp32, 128, profiling)
        self.assertIn('"replay-selected", provider, "compare", 1, 0, 0', controller)
        self.assertIn('"replay-original-after", original, "compare", 0, 0, 0', controller)
        self.assertIn("or int(row[5]) != profiling", controller)
        self.assertIn('"native-full-v3"', controller)
        self.assertIn("math.isfinite(event.get", controller)
        cpp = native_source((ROOT / "dense_long_k_fixed_native_v2.cpp").read_text())
        self.assertIn('if (profiling == "1") options.EnableProfiling(prefix.c_str());', cpp)
        self.assertIn("Explicit native profiling selector must be 0 or 1", cpp)
        self.assertIn('" profiling=" << profiling', cpp)
        self.assertIn("std::memcmp(&expected[i], &observed[i], sizeof(expected[i]))", cpp)


if __name__ == "__main__":
    unittest.main()
