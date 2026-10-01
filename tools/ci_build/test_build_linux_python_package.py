# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Exercise Linux wheel collection across Python ABIs without compiling native code."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path

_CI_BUILD_DIR = Path(__file__).resolve().parent
_BUILD_SCRIPT = _CI_BUILD_DIR / "github/linux/build_linux_python_package.sh"
_ABI_TAGS = ("cp311", "cp313t", "cp312")


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("bash"), "Requires Linux and Bash")
class TestLinuxPythonPackage(unittest.TestCase):
    def _run_build(self, *, unified: bool):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            build_dir = root / "build"
            interpreters = root / "python"
            # Stub compilation and pip, but run the real shell script, filesystem operations, and merger.
            fake_python = f"#!{sys.executable}\n" + textwrap.dedent(
                f"""\
                import os
                from pathlib import Path
                import sys

                sys.path.insert(0, {str(_CI_BUILD_DIR)!r})
                from test_merge_python_wheels import write_wheel

                if sys.argv[1:3] == ["-m", "pip"]:
                    sys.exit(0)
                if Path(sys.argv[1]).name != "build.py":
                    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])

                abi_tag = Path(sys.argv[0]).parents[1].name.split("-")[1]
                build_dir = Path(sys.argv[sys.argv.index("--build_dir") + 1])
                config = sys.argv[sys.argv.index("--config") + 1]
                capi = build_dir / config / "onnxruntime/capi"
                capi.mkdir(parents=True, exist_ok=True)
                (capi / "libonnxruntime_providers_shared.so").write_bytes(b"shared native library")
                if "--build_wheel" in sys.argv:
                    dist = build_dir / config / "dist"
                    dist.mkdir(exist_ok=True)
                    wheel = Path(write_wheel(str(dist), abi_tag=abi_tag, platform_tag="linux_x86_64"))
                    # setup.py selects the first *linux*.whl for auditwheel. A stale manylinux
                    # wheel also matches, so require an unambiguous current-ABI candidate.
                    candidates = list(dist.glob("*linux*.whl"))
                    if candidates != [wheel]:
                        sys.exit("Stale wheels make auditwheel's input ambiguous: " + str(candidates))
                    write_wheel(str(dist), abi_tag=abi_tag, platform_tag="manylinux_2_28_x86_64")
                    wheel.unlink()
                """
            )
            for abi_tag in _ABI_TAGS:
                python_tag = abi_tag.removesuffix("t")
                executable = interpreters / f"{python_tag}-{abi_tag}" / "bin" / f"python3.{python_tag[3:]}"
                executable.parent.mkdir(parents=True)
                executable.write_text(fake_python)
                executable.chmod(0o755)

            # Redirect the container's absolute paths into the temporary directory.
            script = _BUILD_SCRIPT.read_text()
            script = script.replace("/opt/python/", f"{interpreters}/")
            script = script.replace("/onnxruntime_src/", f"{_CI_BUILD_DIR.parents[1]}/")
            script = script.replace("/build/", f"{build_dir}/").replace('"/build"', f'"{build_dir}"')
            script_path = root / "build.sh"
            script_path.write_text(script)

            # Avoid modifying a developer's ccache statistics.
            mock_bin = root / "bin"
            mock_bin.mkdir()
            ccache = mock_bin / "ccache"
            ccache.write_text("#!/bin/sh\nexit 0\n")
            ccache.chmod(0o755)
            command = ["bash", str(script_path), "-d", "CPU", "-c", "Release"]
            if unified:
                command.append("-m")
            result = subprocess.run(
                command,
                env={**os.environ, "PATH": f"{mock_bin}{os.pathsep}{os.environ['PATH']}"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            wheels = list((build_dir / "dist").glob("*.whl"))
            self.assertEqual(len(wheels), 1 if unified else len(_ABI_TAGS))
            tags = []
            for wheel in wheels:
                with zipfile.ZipFile(wheel) as archive:
                    metadata = archive.read("onnxruntime-1.28.0.dist-info/WHEEL").decode()
                    tags.extend(line for line in metadata.splitlines() if line.startswith("Tag:"))
            self.assertCountEqual(
                tags,
                [f"Tag: {abi.removesuffix('t')}-{abi}-manylinux_2_28_x86_64" for abi in _ABI_TAGS],
            )

    def test_unified_build_repairs_and_retains_every_abi(self):
        self._run_build(unified=True)

    def test_per_python_build_retains_separate_repaired_wheels(self):
        self._run_build(unified=False)


if __name__ == "__main__":
    unittest.main()
