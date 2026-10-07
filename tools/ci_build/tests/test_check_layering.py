# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_layering


class LayeringCheckerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "onnxruntime/core").mkdir(parents=True)
        (self.root / "include/onnxruntime/core").mkdir(parents=True)

    def write(self, name, text=""):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_quoted_angled_multiline_and_deduplicated_edges(self):
        source = "onnxruntime/core/common/source.cc"
        self.write(
            source,
            '#include "core/framework/one.h"\n'
            "# include <core/framework/two.h>\n"
            "#include \\\n<core/framework/three.h>\n"
            '#include /* rationale */ "core/framework/one.h"\n',
        )
        self.assertEqual(
            check_layering.find_upward_includes(self.root),
            {(source, f"core/framework/{name}.h") for name in ("one", "two", "three")},
        )

    def test_comments_raw_strings_downward_unknown_and_top_sources_excluded(self):
        self.write(
            "onnxruntime/core/framework/source.cc",
            '#include "core/graph/down.h"\n'
            '#include "core/framework/same.h"\n'
            '#include "core/unknown/unknown.h"\n'
            '/*\n#include "core/session/comment.h"\n*/\n'
            '// #include "core/session/comment2.h"\n'
            'const char* s = R"tag(\n#include "core/session/string.h"\n)tag";\n',
        )
        self.write("onnxruntime/core/providers/source.cc", '#include "core/session/accepted-glue.h"\n')
        self.assertEqual(check_layering.find_upward_includes(self.root), set())

    def test_public_api_exemption_and_public_source_scan(self):
        self.write("include/onnxruntime/core/session/public.h")
        self.write("onnxruntime/core/common/source.cc", "#include <core/session/public.h>\n")
        source = "include/onnxruntime/core/graph/source.h"
        self.write(source, '#include "core/framework/private.h"\n')
        self.assertEqual(check_layering.find_upward_includes(self.root), {(source, "core/framework/private.h")})

    def test_new_resolved_exact_baseline_and_update(self):
        edge = ("onnxruntime/core/graph/source.cc", "core/framework/private.h")
        for hits, baseline, expected in [({edge}, set(), 1), (set(), {edge}, 1), ({edge}, {edge}, 0)]:
            with (
                mock.patch.object(sys, "argv", ["check_layering"]),
                mock.patch.object(check_layering, "find_upward_includes", return_value=hits),
                mock.patch.object(check_layering, "_BASELINE", frozenset(baseline)),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(check_layering.main(), expected)
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["check_layering", "--update-baseline"]),
            mock.patch.object(check_layering, "find_upward_includes", return_value={edge}),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(check_layering.main(), 0)
        self.assertIn(repr(edge[0]), output.getvalue())

    def test_missing_tree_and_read_failure_raise(self):
        with self.assertRaises(NotADirectoryError):
            check_layering.find_upward_includes(self.root / "missing")
        self.write("onnxruntime/core/graph/source.cc", '#include "core/framework/private.h"\n')
        with (
            mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")),
            self.assertRaises(PermissionError),
        ):
            check_layering.find_upward_includes(self.root)
        with (
            mock.patch.object(sys, "argv", ["check_layering"]),
            mock.patch.object(check_layering, "find_upward_includes", side_effect=PermissionError("denied")),
            contextlib.redirect_stderr(io.StringIO()) as output,
        ):
            self.assertEqual(check_layering.main(), 2)
        self.assertIn("cannot scan include layering: denied", output.getvalue())

    def test_directory_failure_raises(self):
        def failed_walk(_root, *, onerror):
            onerror(PermissionError("directory denied"))

        with (
            mock.patch.object(check_layering.os, "walk", side_effect=failed_walk),
            self.assertRaises(PermissionError),
        ):
            check_layering.find_upward_includes(self.root)


if __name__ == "__main__":
    unittest.main()
