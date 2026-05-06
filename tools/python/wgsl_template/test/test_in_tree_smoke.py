"""Smoke tests against the real in-tree wgsl.template files.

These three templates back the Pad, Transpose, and im2col-matmul
operators. The smoke test only verifies that the build completes
cleanly and produces the expected file structure. The strongest
end-to-end signal — operator unit tests passing on a real GPU — is a
manual step performed before opening the PR.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from wgsl_template import build  # noqa: E402

# Repo root: tools/python/wgsl_template/test/ -> ../../../..
_REPO_ROOT = _THIS_DIR.parent.parent.parent.parent
# WebGPU EP source root, where the real templates live.
_WEBGPU_ROOT = _REPO_ROOT / "onnxruntime" / "core" / "providers" / "webgpu"

_EXPECTED_TEMPLATES = {
    "nn/im2col_matmul.wgsl.template",
    "tensor/oihw_to_ohwi.wgsl.template",
    "tensor/pad.wgsl.template",
}


class InTreeTemplatesSmokeTest(unittest.TestCase):
    def _run(self, generator: str) -> Path:
        out = Path(tempfile.mkdtemp(prefix="wgsl_smoke_"))
        try:
            build(
                source_dirs=[_WEBGPU_ROOT],
                out_dir=out,
                generator=generator,
                include_path_prefix="wgsl_template_gen/",
                preserve_code_reference=True,
            )
        except Exception:
            shutil.rmtree(out, ignore_errors=True)
            raise
        return out

    def test_static_cpp_literal_builds(self) -> None:
        out = self._run("static-cpp-literal")
        try:
            self.assertTrue((out / "index.h").is_file())
            self.assertTrue((out / "index_impl.h").is_file())
            # Literal mode does not emit string_table.h.
            self.assertFalse((out / "string_table.h").exists())
            for tpl in _EXPECTED_TEMPLATES:
                base = tpl[: -len(".wgsl.template")]
                self.assertTrue(
                    (out / "generated" / f"{base}.h").is_file(),
                    f"missing generated/{base}.h",
                )
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_static_cpp_builds(self) -> None:
        out = self._run("static-cpp")
        try:
            self.assertTrue((out / "index.h").is_file())
            self.assertTrue((out / "index_impl.h").is_file())
            # Release mode emits string_table.h.
            self.assertTrue((out / "string_table.h").is_file())
            for tpl in _EXPECTED_TEMPLATES:
                base = tpl[: -len(".wgsl.template")]
                self.assertTrue(
                    (out / "generated" / f"{base}.h").is_file(),
                    f"missing generated/{base}.h",
                )
            # Sanity: index.h should reference each template by its full name.
            index_text = (out / "index.h").read_text(encoding="utf-8")
            for tpl in _EXPECTED_TEMPLATES:
                self.assertIn(tpl, index_text)
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
