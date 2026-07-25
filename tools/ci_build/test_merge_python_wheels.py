# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Unit tests for merge_python_wheels.py.

Run them with:  python -m unittest tools.ci_build.test_merge_python_wheels
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_python_wheels.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("merge_python_wheels", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merger = _load_script()

# Files that every input wheel shares byte for byte; this is where the native libraries live.
SHARED_FILES = {
    "onnxruntime/__init__.py": b"# onnxruntime\n",
    "onnxruntime/capi/_pybind_state.py": b"from .onnxruntime_pybind11_state import *  # noqa\n",
    "onnxruntime/capi/libonnxruntime_providers_shared.so": b"native library, identical everywhere",
}


def write_wheel(
    directory: str,
    *,
    name: str = "onnxruntime",
    version: str = "1.28.0",
    abi_tag: str = "cp312",
    platform_tag: str = "manylinux_2_28_x86_64",
    pybind_suffix: str = ".so",
    extra_files: dict[str, bytes] | None = None,
) -> str:
    """Write a realistic single-CPython wheel and return its path."""
    # A free-threaded wheel keeps the plain Python tag: cp313-cp313t-<platform>.
    python_tag = abi_tag.removesuffix("t")
    path = os.path.join(directory, f"{name}-{version}-{python_tag}-{abi_tag}-{platform_tag}.whl")
    dist_info = f"{name}-{version}.dist-info"

    members = dict(SHARED_FILES)
    # The binding is the one file that legitimately differs between interpreters.
    members[f"onnxruntime/capi/onnxruntime_pybind11_state{pybind_suffix}"] = f"binding for {abi_tag}".encode()
    members.update(extra_files or {})
    members[f"{dist_info}/METADATA"] = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
    members[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: bdist_wheel\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {python_tag}-{abi_tag}-{platform_tag}\n"
    ).encode()

    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for member, data in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
        writer.writerow([member, "sha256=" + digest, len(data)])
    writer.writerow([f"{dist_info}/RECORD", "", ""])
    members[f"{dist_info}/RECORD"] = record.getvalue().encode()

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, data in sorted(members.items()):
            archive.writestr(member, data)
    return path


def run_merger(out_dir: str, wheels: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, _SCRIPT, "--out-dir", out_dir, *wheels],
        capture_output=True,
        text=True,
        check=False,
    )


class TestExtensionSuffix(unittest.TestCase):
    def test_linux(self):
        self.assertEqual(
            merger.extension_suffix("cp312", "manylinux_2_28_x86_64"),
            ".cpython-312-x86_64-linux-gnu.so",
        )

    def test_linux_aarch64(self):
        self.assertEqual(
            merger.extension_suffix("cp313", "manylinux_2_28_aarch64"),
            ".cpython-313-aarch64-linux-gnu.so",
        )

    def test_linux_free_threaded(self):
        self.assertEqual(
            merger.extension_suffix("cp313t", "manylinux_2_28_x86_64"),
            ".cpython-313t-x86_64-linux-gnu.so",
        )

    def test_windows(self):
        self.assertEqual(merger.extension_suffix("cp312", "win_amd64"), ".cp312-win_amd64.pyd")
        self.assertEqual(merger.extension_suffix("cp314t", "win_amd64"), ".cp314t-win_amd64.pyd")

    def test_macos(self):
        self.assertEqual(merger.extension_suffix("cp312", "macosx_11_0_arm64"), ".cpython-312-darwin.so")

    def test_rejects_non_cpython_abi(self):
        with self.assertRaises(ValueError):
            merger.extension_suffix("abi3", "win_amd64")

    def test_rejects_unknown_platform(self):
        with self.assertRaises(ValueError):
            merger.extension_suffix("cp312", "some_future_platform")


class TestIsPybindModule(unittest.TestCase):
    def test_accepts_the_binding(self):
        self.assertTrue(merger.is_pybind_module("onnxruntime/capi/onnxruntime_pybind11_state.so"))
        self.assertTrue(merger.is_pybind_module("onnxruntime/capi/onnxruntime_pybind11_state.pyd"))
        self.assertTrue(
            merger.is_pybind_module("onnxruntime/capi/onnxruntime_pybind11_state.cpython-312-x86_64-linux-gnu.so")
        )

    def test_rejects_other_native_libraries(self):
        self.assertFalse(merger.is_pybind_module("onnxruntime/capi/libonnxruntime_providers_cuda.so"))
        self.assertFalse(merger.is_pybind_module("onnxruntime/capi/libonnxruntime.so.1.28.0"))

    def test_rejects_the_binding_outside_capi(self):
        self.assertFalse(merger.is_pybind_module("onnxruntime/onnxruntime_pybind11_state.so"))


class TestMergedPythonTags(unittest.TestCase):
    def test_collapses_the_free_threaded_abi(self):
        self.assertEqual(merger.merged_python_tags(["cp313", "cp313t"]), ["cp313"])

    def test_sorts_and_deduplicates(self):
        tags = merger.merged_python_tags(["cp314", "cp312", "cp313", "cp313t", "cp314t"])
        self.assertEqual(tags, ["cp312", "cp313", "cp314"])

    def test_older_versions_need_no_free_threaded_build(self):
        self.assertEqual(merger.merged_python_tags(["cp310", "cp311", "cp312"]), ["cp310", "cp311", "cp312"])

    def test_rejects_a_missing_free_threaded_build(self):
        with self.assertRaises(SystemExit) as caught:
            merger.merged_python_tags(["cp312", "cp313"])
        self.assertIn("cp313 is given but cp313t is not", str(caught.exception))

    def test_rejects_a_missing_default_build(self):
        with self.assertRaises(SystemExit) as caught:
            merger.merged_python_tags(["cp314t"])
        self.assertIn("cp314t is given but cp314 is not", str(caught.exception))


class TestMergeRejects(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")

    def test_rejects_a_single_wheel(self):
        wheel = write_wheel(self.tmp.name, abi_tag="cp312")
        result = run_merger(self.out, [wheel])
        self.assertEqual(result.returncode, 1)
        self.assertIn("at least two wheels are required", result.stderr)

    def test_rejects_different_platforms(self):
        """x86_64 and aarch64 share no native library and must stay separate packages."""
        wheels = [
            write_wheel(self.tmp.name, abi_tag="cp312", platform_tag="manylinux_2_28_x86_64"),
            write_wheel(self.tmp.name, abi_tag="cp313", platform_tag="manylinux_2_28_aarch64"),
        ]
        result = run_merger(self.out, wheels)
        self.assertEqual(result.returncode, 1)
        self.assertIn("must share name, version and platform tag", result.stderr)

    def test_rejects_windows_mixed_with_linux(self):
        wheels = [
            write_wheel(self.tmp.name, abi_tag="cp312", platform_tag="win_amd64", pybind_suffix=".pyd"),
            write_wheel(self.tmp.name, abi_tag="cp313", platform_tag="manylinux_2_28_x86_64"),
        ]
        result = run_merger(self.out, wheels)
        self.assertEqual(result.returncode, 1)
        self.assertIn("must share name, version and platform tag", result.stderr)

    def test_rejects_different_versions(self):
        wheels = [
            write_wheel(self.tmp.name, abi_tag="cp312", version="1.28.0"),
            write_wheel(self.tmp.name, abi_tag="cp313", version="1.29.0"),
        ]
        result = run_merger(self.out, wheels)
        self.assertEqual(result.returncode, 1)
        self.assertIn("must share name, version and platform tag", result.stderr)

    def test_rejects_different_distributions(self):
        wheels = [
            write_wheel(self.tmp.name, abi_tag="cp312", name="onnxruntime"),
            write_wheel(self.tmp.name, abi_tag="cp313", name="onnxruntime_gpu"),
        ]
        result = run_merger(self.out, wheels)
        self.assertEqual(result.returncode, 1)
        self.assertIn("must share name, version and platform tag", result.stderr)

    def test_rejects_a_repeated_abi_tag(self):
        first = write_wheel(self.tmp.name, abi_tag="cp312")
        copy = os.path.join(self.tmp.name, "copy")
        os.makedirs(copy)
        second = write_wheel(copy, abi_tag="cp312")
        result = run_merger(self.out, [first, second])
        self.assertEqual(result.returncode, 1)
        self.assertIn("the same ABI tag appears twice", result.stderr)

    def test_rejects_another_differing_native_library(self):
        """A provider library that differs would be renamed into a file ORT never loads."""
        wheels = [
            write_wheel(
                self.tmp.name,
                abi_tag="cp311",
                extra_files={"onnxruntime/capi/libonnxruntime_providers_cuda.so": b"cuda build A"},
            ),
            write_wheel(
                self.tmp.name,
                abi_tag="cp312",
                extra_files={"onnxruntime/capi/libonnxruntime_providers_cuda.so": b"cuda build B"},
            ),
        ]
        result = run_merger(self.out, wheels)
        self.assertEqual(result.returncode, 1)
        self.assertIn("only the onnxruntime pybind11 extension module may differ", result.stderr)
        self.assertIn("libonnxruntime_providers_cuda.so", result.stderr)

    def test_rejects_a_differing_python_file(self):
        wheels = [
            write_wheel(self.tmp.name, abi_tag="cp311", extra_files={"onnxruntime/version.py": b"A"}),
            write_wheel(self.tmp.name, abi_tag="cp312", extra_files={"onnxruntime/version.py": b"B"}),
        ]
        result = run_merger(self.out, wheels)
        self.assertEqual(result.returncode, 1)
        self.assertIn("only the onnxruntime pybind11 extension module may differ", result.stderr)
        self.assertIn("onnxruntime/version.py", result.stderr)

    def test_rejects_a_missing_free_threaded_wheel(self):
        wheels = [
            write_wheel(self.tmp.name, abi_tag="cp313"),
            write_wheel(self.tmp.name, abi_tag="cp314"),
        ]
        result = run_merger(self.out, wheels)
        self.assertEqual(result.returncode, 1)
        self.assertIn("cp313 is given but cp313t is not", result.stderr)


class TestMergeSucceeds(unittest.TestCase):
    PLATFORM = "manylinux_2_28_x86_64"
    ABI_TAGS = ("cp312", "cp313", "cp313t", "cp314", "cp314t")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")
        self.wheels = [
            write_wheel(self.tmp.name, abi_tag=abi_tag, platform_tag=self.PLATFORM) for abi_tag in self.ABI_TAGS
        ]
        result = run_merger(self.out, self.wheels)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.merged = os.path.join(self.out, os.listdir(self.out)[0])

    def test_file_name_lists_only_the_bundled_interpreters(self):
        self.assertEqual(
            os.path.basename(self.merged),
            f"onnxruntime-1.28.0-cp312.cp313.cp314-none-{self.PLATFORM}.whl",
        )

    def test_wheel_metadata_carries_every_tag(self):
        with zipfile.ZipFile(self.merged) as archive:
            wheel_text = archive.read("onnxruntime-1.28.0.dist-info/WHEEL").decode()
        tags = sorted(line for line in wheel_text.splitlines() if line.startswith("Tag:"))
        self.assertEqual(
            tags,
            [f"Tag: cp{minor}-none-{self.PLATFORM}" for minor in ("312", "313", "314")],
        )
        # The wheel must not claim interpreters it has no binding for.
        self.assertNotIn("py3-none", wheel_text)
        self.assertIn("Root-Is-Purelib: false", wheel_text)

    def test_every_binding_is_stored_under_its_abi_suffix(self):
        with zipfile.ZipFile(self.merged) as archive:
            names = set(archive.namelist())
        for abi_tag in self.ABI_TAGS:
            suffix = merger.extension_suffix(abi_tag, self.PLATFORM)
            member = f"onnxruntime/capi/onnxruntime_pybind11_state{suffix}"
            self.assertIn(member, names)
            with zipfile.ZipFile(self.merged) as archive:
                self.assertEqual(archive.read(member), f"binding for {abi_tag}".encode())
        # The unsuffixed name would shadow the ABI specific ones, so it must be gone.
        self.assertNotIn("onnxruntime/capi/onnxruntime_pybind11_state.so", names)

    def test_shared_files_are_stored_once(self):
        with zipfile.ZipFile(self.merged) as archive:
            names = archive.namelist()
        for shared in SHARED_FILES:
            self.assertEqual(names.count(shared), 1, shared)

    def test_record_matches_the_archive(self):
        with zipfile.ZipFile(self.merged) as archive:
            names = set(archive.namelist())
            record_text = archive.read("onnxruntime-1.28.0.dist-info/RECORD").decode()
            rows = list(csv.reader(io.StringIO(record_text)))
            self.assertEqual({row[0] for row in rows}, names)
            for member, digest, size in rows:
                if member.endswith("dist-info/RECORD"):
                    self.assertEqual((digest, size), ("", ""))
                    continue
                data = archive.read(member)
                expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
                self.assertEqual(digest, "sha256=" + expected, member)
                self.assertEqual(int(size), len(data), member)

    def test_the_merged_wheel_is_smaller_than_the_inputs(self):
        total_in = sum(os.path.getsize(wheel) for wheel in self.wheels)
        self.assertLess(os.path.getsize(self.merged), total_in)


class TestMergeSucceedsOnWindows(unittest.TestCase):
    def test_windows_bindings_keep_the_pyd_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            wheels = [
                write_wheel(tmp, abi_tag=abi_tag, platform_tag="win_amd64", pybind_suffix=".pyd")
                for abi_tag in ("cp312", "cp314", "cp314t")
            ]
            result = run_merger(out, wheels)
            self.assertEqual(result.returncode, 0, result.stderr)

            merged = os.path.join(out, os.listdir(out)[0])
            self.assertEqual(os.path.basename(merged), "onnxruntime-1.28.0-cp312.cp314-none-win_amd64.whl")
            with zipfile.ZipFile(merged) as archive:
                names = set(archive.namelist())
            for abi_tag in ("cp312", "cp314", "cp314t"):
                self.assertIn(f"onnxruntime/capi/onnxruntime_pybind11_state.{abi_tag}-win_amd64.pyd", names)


if __name__ == "__main__":
    unittest.main()
