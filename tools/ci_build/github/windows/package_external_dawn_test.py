import unittest

from package_external_dawn import _PRIVATE_INCLUDES, _STANDALONE_SUPPORT, standalone_proc


class StandaloneProcTest(unittest.TestCase):
    def setUp(self):
        self.source = (
            '#include "dawn/dawn_proc.h"\n' + "\n".join(sorted(_PRIVATE_INCLUDES)) + "\nvoid dawnProcSetProcs() {}\n"
        )

    def test_known_private_includes_are_replaced(self):
        result = standalone_proc(self.source)
        self.assertEqual(result.count(_STANDALONE_SUPPORT), 1)
        self.assertIn('#include "dawn/dawn_proc.h"', result)
        self.assertIn("void dawnProcSetProcs() {}", result)
        for include in _PRIVATE_INCLUDES:
            self.assertNotIn(include, result)

    def test_missing_private_include_is_rejected(self):
        for include in _PRIVATE_INCLUDES:
            with self.subTest(include=include), self.assertRaises(ValueError):
                standalone_proc(self.source.replace(include, ""))

    def test_added_private_include_is_rejected(self):
        for include in (
            '#include "src/utils/new_private_header.h"',
            '  # include "src/utils/new_private_header.h"',
            "#include <src/utils/new_private_header.h>",
        ):
            with self.subTest(include=include), self.assertRaisesRegex(ValueError, "unexpected:"):
                standalone_proc(self.source + include + "\n")

    def test_spaced_known_include_is_replaced(self):
        source = self.source.replace('#include "src/', '  # include "src/')
        result = standalone_proc(source)
        self.assertEqual(result.count(_STANDALONE_SUPPORT), 1)
        self.assertNotIn("src/", result)


if __name__ == "__main__":
    unittest.main()
