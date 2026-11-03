import unittest

from attribute_cpu_libc_ifunc_v1 import ifunc_ranges, libc_base


class IfuncTests(unittest.TestCase):
    def test_only_exact_entry_fdes_and_preserve_aliases(self):
        entries = [
            {"name": name, "elf_address": address, "inside_verified_libc": True}
            for name, address in (("memcpy", 0x100), ("memmove", 0x100),
                                  ("inside_not_entry", 0x110))
        ]
        ranges, unmatched = ifunc_ranges(
            entries, "000 FDE cie=0 pc=00000100..00000200\n")
        self.assertEqual(ranges, [
            {"begin": 0x100, "end": 0x200, "aliases": ["memcpy", "memmove"]}])
        self.assertEqual([row["name"] for row in unmatched], ["inside_not_entry"])

    def test_mapping_missing_or_conflicting_rejects(self):
        line = (
            "PERF_RECORD_MMAP2 1/1: [0x1000(0x1000) @ 0x100 "
            "08:01 1 0]: r-xp /usr/lib/x86_64-linux-gnu/libc.so.6"
        )
        self.assertEqual(libc_base([line]), (0xF00, [(0x1000, 0x2000)]))
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            libc_base([line, line.replace("[0x1000", "[0x2000")])
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            libc_base([])


if __name__ == "__main__":
    unittest.main()
