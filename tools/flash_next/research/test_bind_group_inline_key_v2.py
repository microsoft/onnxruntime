import unittest

import build_bind_group_inline_key as original
import build_bind_group_inline_key_v2 as current


class BindGroupInlineKeyV2Tests(unittest.TestCase):
    def test_key_and_hash_helpers_are_unchanged(self):
        self.assertEqual(current.HELPERS, original.HELPERS)
        self.assertEqual(current.HASH, original.HASH)

    def test_invalid_selector_is_checked_before_program_work(self):
        source = current.SOURCE
        host = str(source).replace("/workspace/", "/home/sayanshaw/repos/", 1)
        from pathlib import Path
        result = current.transform(Path(host).read_text())
        signature = "Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {"
        self.assertIn(signature + "\n  (void)BindGroupInlineKeyEnabled();", result)
        self.assertLess(result.index("bool BindGroupInlineKeyEnabled()"), result.index(signature))
        self.assertIn("cache.entries.find(lookup_key)", result)
        self.assertIn("key.assign(lookup_key.begin(), lookup_key.end())", result)


if __name__ == "__main__":
    unittest.main()
