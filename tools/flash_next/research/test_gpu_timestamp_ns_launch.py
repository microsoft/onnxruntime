import unittest

from run_gpu_timestamp_ns_v1 import validate_profile


class GpuTimestampNativeScopeTests(unittest.TestCase):
    def events(self, precise):
        args = {"cache_key": "HCUpPreMixPacked2F16:128,1,1:shader", "shapes": "frozen actual shapes"}
        if precise:
            args["gpu_duration_ns"] = "12741"
        return [{"cat": "Kernel", "name": "&HCUpPreMixF16&HCUpPreMixPacked2F16",
                 "pid": -1, "tid": -1, "ph": "X", "ts": index * 100, "dur": 13,
                 "args": args.copy()} for index in range(288)]

    def test_original_compute_scope_and_actual_sub_microsecond_evidence_are_required(self):
        self.assertFalse(validate_profile(self.events(False), False)["nanosecond_metadata"])
        precise = validate_profile(self.events(True), True)
        self.assertEqual(precise["fractional_microsecond_events"], 288)
        self.assertEqual(precise["minimum_gpu_us"], 12.741)
        with self.assertRaises(RuntimeError):
            validate_profile(self.events(True), False)
        events = self.events(True)
        for event in events:
            event["args"]["gpu_duration_ns"] = "13000"
        with self.assertRaises(RuntimeError):
            validate_profile(events, True)

    def test_wrong_shader_geometry_scope_and_silent_legacy_fallback_reject(self):
        for key, value in (("name", "&HCUpPreMixF16&HCUpPreMixPacked160F16"), ("pid", 1),
                           ("args", {"cache_key": "key:160,1,1:", "shapes": "wrong"})):
            events = self.events(True)
            events[0][key] = value
            with self.assertRaises(RuntimeError):
                validate_profile(events, True)
        with self.assertRaises(RuntimeError):
            validate_profile(self.events(True)[:-1], True)
        with self.assertRaises(ValueError):
            validate_profile(self.events(False), True)


if __name__ == "__main__":
    unittest.main()
