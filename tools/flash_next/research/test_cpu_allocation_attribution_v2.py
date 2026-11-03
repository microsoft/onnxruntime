import unittest

from attribute_cpu_allocations_v2 import allocation_samples


class AllocationAttributionTests(unittest.TestCase):
    def test_operator_new_is_not_the_actionable_allocation_site(self):
        text = """python3 42 100.500000: 5000000 cpu-clock:u:
 aa malloc+0x98 (/lib/libc.so.6)
 bb operator new(unsigned long)+0x1b (/opt/conda/lib/libstdc++.so.6)
 cc absl::PrepareInsert()+0x44 (/workspace/provider.so)
 dd dawn::native::SyncScopeUsageTracker::BufferUsedAs()+0x12 (/workspace/provider.so)
 ee onnxruntime::webgpu::DispatchCommand()+0x1 (/workspace/provider.so)
"""
        count, records = allocation_samples(text, (100.0, 101.0))
        self.assertEqual(count, 1)
        self.assertEqual(records[0]["allocation_site"], "absl::PrepareInsert()")
        self.assertEqual(records[0]["dawn_ancestor"],
                         "dawn::native::SyncScopeUsageTracker::BufferUsedAs()")

    def test_missing_caller_remains_explicit(self):
        text = """python3 42 100.500000: 5000000 cpu-clock:u:
 aa malloc+0x98 (/lib/libc.so.6)
"""
        self.assertIsNone(allocation_samples(text, (100.0, 101.0))[1][0]["allocation_site"])

    def test_rejects_outside_marker_sample(self):
        with self.assertRaisesRegex(RuntimeError, "outside"):
            allocation_samples("python3 42 99.000000: 5000000 cpu-clock:u:", (100.0, 101.0))

    def test_does_not_count_nonallocation_sample(self):
        text = """python3 42 100.500000: 5000000 cpu-clock:u:
 aa onnxruntime::Run()+0x1 (/workspace/provider.so)
"""
        self.assertEqual(allocation_samples(text, (100.0, 101.0)), (1, []))


if __name__ == "__main__":
    unittest.main()
