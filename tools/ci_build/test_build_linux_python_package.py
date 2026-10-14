# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Exercise Linux wheel collection across Python ABIs without compiling native code."""

from __future__ import annotations

import os
import re
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
PYTHON_PATHS = (
    "cp311-cp311/bin/python3.11",
    "cp313-cp313/bin/python3.13",
    "cp313-cp313t/bin/python3.13",
    "cp314-cp314/bin/python3.14",
    "cp314-cp314t/bin/python3.14",
    "cp315-cp315/bin/python3.15",
    "cp312-cp312/bin/python3.12",
)


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
                # Like build.py, configure without Python bindings unless they are requested.
                if "--update" in sys.argv and not {{"--build_wheel", "--enable_pybind"}} & set(sys.argv):
                    sys.exit("onnxruntime_ENABLE_PYTHON=OFF would drop the onnxruntime_pybind11_state target")

                abi_tag = Path(sys.argv[0]).parents[1].name.split("-")[1]
                build_dir = Path(sys.argv[sys.argv.index("--build_dir") + 1])
                config = sys.argv[sys.argv.index("--config") + 1]
                if "--update" in sys.argv:
                    # Like FindPython, a cached interpreter is not re-detected when Python_EXECUTABLE changes.
                    cache = build_dir / config / "CMakeCache.txt"
                    entry = f"_Python_EXECUTABLE:INTERNAL={{sys.argv[0]}}\\n"
                    if cache.exists() and cache.read_text() not in ("", entry):
                        sys.exit("Stale FindPython cache would drop Python::NumPy: " + cache.read_text())
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(entry)
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


@unittest.skipUnless(Path("/bin/bash").is_file(), "Requires Bash")
class BuildLinuxPythonPackageTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.build_dir = self.root / "build"
        self.command_log = self.root / "commands.log"
        source_root = Path(__file__).resolve().parents[2]
        script = source_root / "tools/ci_build/github/linux/build_linux_python_package.sh"
        self.script = self.root / "build.sh"
        self.script.write_text(
            script.read_text()
            .replace("/build/", f"{self.build_dir}/")
            .replace('"/build"', f'"{self.build_dir}"')
            .replace("/onnxruntime_src", str(source_root))
        )
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        # Stop at the first external command so rejected selectors cannot mutate or install anything.
        for command in ("mkdir", "ccache", "uname", "mktemp", "grep", "rm", "ls", "cp"):
            stub = bin_dir / command
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$0 $*" >> "$COMMAND_LOG"\nexit 73\n')
            stub.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "COMMAND_LOG": str(self.command_log)}

    def run_script(self, python_exe=None):
        return subprocess.run(
            ["/bin/bash", str(self.script), "-d", "CPU", "-c", "Release"]
            + (["-p", str(python_exe)] if python_exe is not None else []),
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def assert_rejected_before_external_commands(self, python_exe):
        result = self.run_script(python_exe)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("ERROR: Requested Python executable", result.stdout)
        self.assertIn(str(python_exe), result.stdout)
        self.assertFalse(self.command_log.exists())
        self.assertFalse(self.build_dir.exists())

    def prepare_default_interpreters(self, missing_python315=False):
        self.script.write_text(self.script.read_text().replace("/opt/python/", f"{self.root}/python/"))
        for relative_path in PYTHON_PATHS:
            if missing_python315 and relative_path.startswith("cp315-"):
                continue
            executable = self.root / "python" / relative_path
            executable.parent.mkdir(parents=True)
            executable.write_text('#!/bin/bash\nprintf "%s|%s\\n" "$0" "$*" >> "$COMMAND_LOG"\n')
            executable.chmod(0o755)
        bin_dir = self.root / "bin"
        for command in ("mkdir", "uname", "mktemp", "grep", "rm", "ls"):
            (bin_dir / command).unlink()
        # Interpreter stubs record build requests; copying their nonexistent wheels is a no-op.
        for command in ("ccache", "cp"):
            (bin_dir / command).write_text("#!/bin/bash\nexit 0\n")

    def assert_default_calls(self, expected):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [line.split("|", 1) for line in self.command_log.read_text().splitlines()]
        self.assertEqual(
            [path for path, _ in calls], [str(self.root / "python" / p) for p in expected for _ in range(2)]
        )
        for pip_call, build_call in zip(calls[::2], calls[1::2], strict=True):
            self.assertTrue(pip_call[1].startswith("-m pip install -r "), pip_call)
            if "cp315-cp315/" in pip_call[0]:
                source_root = Path(__file__).resolve().parents[2]
                self.assertEqual(
                    pip_call[1], f"-m pip install -r {source_root}/tools/ci_build/github/linux/python/requirements.txt"
                )
            self.assertIn("/tools/ci_build/build.py --build_dir ", build_call[1])
        self.assertTrue(calls[-1][0].endswith("cp312-cp312/bin/python3.12"))
        return result

    def test_default_interpreters_build_in_order_with_python312_last(self):
        self.prepare_default_interpreters()
        self.assert_default_calls(PYTHON_PATHS)

    def test_default_missing_python315_warns_and_continues(self):
        self.prepare_default_interpreters(missing_python315=True)
        result = self.assert_default_calls([p for p in PYTHON_PATHS if not p.startswith("cp315-")])
        self.assertIn(
            f"WARNING: Python executable not found at {self.root}/python/cp315-cp315/bin/python3.15. Skipping this version.",
            result.stdout,
        )

    def test_missing_python315_selector_fails_before_external_commands(self):
        self.assert_rejected_before_external_commands(self.root / "python/cp315-cp315/bin/python3.15")

    def test_missing_selector_fails_before_external_commands(self):
        self.assert_rejected_before_external_commands(self.root / "missing-python")

    def test_non_executable_selector_fails_before_external_commands(self):
        python_exe = self.root / "python"
        python_exe.write_text("#!/bin/bash\nexit 0\n")
        self.assert_rejected_before_external_commands(python_exe)

    def test_valid_selector_invokes_requested_interpreter(self):
        # Permit local harness setup and stop when the selected interpreter receives build arguments.
        bin_dir = self.root / "bin"
        for command in ("mkdir", "uname", "mktemp", "grep", "rm", "ls"):
            (bin_dir / command).unlink()
        (bin_dir / "ccache").write_text("#!/bin/bash\nexit 0\n")
        python_exe = self.root / "selected python"
        python_exe.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\nif [ "$1" == "-m" ]; then exit 0; fi\nexit 73\n'
        )
        python_exe.chmod(0o755)
        result = self.run_script(python_exe)
        self.assertEqual(result.returncode, 73, result.stderr)
        calls = self.command_log.read_text().splitlines()
        source_root = Path(__file__).resolve().parents[2]
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[0], f"-m pip install -r {source_root}/tools/ci_build/github/linux/python/requirements.txt"
        )
        self.assertTrue(calls[1].startswith(f"{source_root}/tools/ci_build/build.py --build_dir {self.build_dir} "))


@unittest.skipUnless(sys.platform.startswith("linux") and Path("/bin/bash").is_file(), "Requires Linux and Bash")
class DependencyInstallerTest(unittest.TestCase):
    def check_dependency_loops(self, missing_python314=False, failing_python314=False):
        source_root = Path(__file__).resolve().parents[2]
        installers = (
            "tools/ci_build/github/linux/docker/scripts/manylinux/install_deps.sh",
            "tools/ci_build/github/linux/docker/inference/aarch64/python/cpu/scripts/install_deps.sh",
        )
        for installer in installers:
            with self.subTest(installer=installer), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = (source_root / installer).read_text()
                shell_options = "\n".join(
                    line for line in source[: source.index("for PYTHON_EXE in")].splitlines() if line.startswith("set ")
                )
                assignment = re.search(r"PYTHON_EXES=\([^\n]+\)", source).group()
                paths = re.findall(r'"(/opt/python/[^"]+)"', assignment)
                expected = [p for p in PYTHON_PATHS if not p.startswith("cp315-")]
                expected.sort()
                if "manylinux" in installer:
                    expected = [p for p in expected if "t/bin" not in p]
                self.assertEqual(paths, [f"/opt/python/{p}" for p in expected])
                loop = source[source.index("for PYTHON_EXE in") : source.index("\ndone") + len("\ndone")]
                for path in paths:
                    if missing_python314 and "cp314-cp314/" in path:
                        continue
                    stub = root / path.removeprefix("/opt/")
                    stub.parent.mkdir(parents=True, exist_ok=True)
                    stub.write_text(
                        '#!/bin/bash\nprintf "%s|%s\\n" "$0" "$*" >> "$COMMAND_LOG"\n'
                        '/bin/cp "$5" "$0.requirements"\n'
                        + ("exit 73\n" if failing_python314 and "cp314-cp314/" in path else "exit 0\n")
                    )
                    stub.chmod(0o755)
                requirements = root / "requirements.txt"
                requirements.write_text("numpy\nonnx==1.23.2\nonnx-ir==0.1.16\nonnxscript==0.6.2\n")
                preparation = ""
                if "manylinux" not in installer:
                    preparation = source[
                        source.index("FREE_THREADED_REQUIREMENTS=") : source.index("for PYTHON_EXE in")
                    ]
                script = root / "install_deps.sh"
                # A successful final command distinguishes errexit from the last pip exit status.
                script.write_text(
                    "#!/bin/bash\n"
                    + shell_options
                    + "\n"
                    + assignment.replace("/opt/python/", f"{root}/python/")
                    + "\n"
                    + preparation
                    + loop
                    + "\n:\n"
                )
                log = root / "commands.log"
                # Pin mktemp's directory so the free-threaded requirements path is predictable.
                temp_dir = root / "tmp"
                temp_dir.mkdir()
                result = subprocess.run(
                    ["/bin/bash", str(script)],
                    cwd=root,
                    env={**os.environ, "COMMAND_LOG": str(log), "TMPDIR": str(temp_dir)},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                calls = [line.split("|", 1) for line in log.read_text().splitlines()]
                stop = paths.index("/opt/python/cp314-cp314/bin/python3.14")
                invoked = paths[: stop + int(failing_python314)] if missing_python314 or failing_python314 else paths
                self.assertEqual([call[0] for call in calls], [str(root / p.removeprefix("/opt/")) for p in invoked])
                self.assertFalse(any("cp315" in str(call) for call in calls))
                if not missing_python314 and not failing_python314:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    if missing_python314:
                        self.assertIn("cp314-cp314/bin/python3.14", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 73)
                for executable, arguments in calls:
                    free_threaded = "t/bin" in executable
                    if free_threaded:
                        self.assertRegex(arguments, rf"^-m pip install -r {re.escape(str(temp_dir))}/\S+( mypy<1.19)?$")
                        self.assertEqual(Path(executable + ".requirements").read_text(), "numpy\n")
                        self.assertEqual(arguments.endswith(" mypy<1.19"), "cp313-cp313t/" in executable)
                    else:
                        requirement_arg = str(requirements) if "manylinux" in installer else "requirements.txt"
                        self.assertEqual(arguments, f"-m pip install -r {requirement_arg}")
                        self.assertEqual(Path(executable + ".requirements").read_text(), requirements.read_text())

    def test_existing_dependency_lists_succeed_without_python315(self):
        self.check_dependency_loops()

    def test_missing_listed_python314_stops_dependency_installation(self):
        self.check_dependency_loops(missing_python314=True)

    def test_python314_pip_failure_stops_dependency_installation(self):
        self.check_dependency_loops(failing_python314=True)


if __name__ == "__main__":
    unittest.main()
