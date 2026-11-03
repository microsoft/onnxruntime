import unittest

from repair_cpu_driver_unwind_v3 import DRIVER, samples, summarize


class DriverUnwindTests(unittest.TestCase):
    def test_header_only_preserves_denominator_and_unknown_libc_is_separate(self):
        text = (
            "python3 12 1.000000: 500 cpu-clock:u:\n\n"
            "python3 12 1.100000: 500 cpu-clock:u:\n"
            "  1234 [unknown] (/usr/lib/x86_64-linux-gnu/libc.so.6)\n\n"
        )
        result = summarize(samples(text, (1.0, 1.2)))
        self.assertEqual(result["strict_samples"], 2)
        self.assertEqual(result["header_only_samples"], 1)
        self.assertEqual(result["unknown_libc_leaf_samples"], 1)
        self.assertEqual(result["driver_stack_first_dawn_ancestor"], [])

    def test_driver_stack_keeps_real_dawn_caller_and_unknown_caller(self):
        text = (
            "python3 12 1.000000: 500 cpu-clock:u:\n"
            f"  1234 [unknown] (/{DRIVER})\n"
            "  5678 dawn::native::Queue::Submit()+0x10 (/workspace/provider.so)\n\n"
            "python3 12 1.100000: 500 cpu-clock:u:\n"
            f"  1234 [unknown] (/{DRIVER})\n\n"
        )
        result = summarize(samples(text, (1.0, 1.2)))
        self.assertEqual(result["unknown_libc_leaf_samples"], 0)
        self.assertEqual(result["driver_stack_first_dawn_ancestor"], [
            {"symbol": "dawn::native::Queue::Submit()", "samples": 1},
            {"symbol": None, "samples": 1},
        ])

    def test_outside_window_and_unparsed_frames_reject(self):
        with self.assertRaisesRegex(RuntimeError, "strict marker"):
            samples("python3 12 2.000000: 500 cpu-clock:u:\n", (1.0, 1.2))
        with self.assertRaisesRegex(RuntimeError, "Unparsed"):
            samples("python3 12 1.000000: 500 cpu-clock:u:\n broken\n", (1.0, 1.2))


if __name__ == "__main__":
    unittest.main()
