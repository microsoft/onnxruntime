# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pathlib
import subprocess
import tempfile
import unittest


@unittest.skipUnless(pathlib.Path("/bin/bash").is_file(), "Requires Bash")
class BuildLinuxPythonPackageTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = pathlib.Path(self.temp_dir.name)
        self.build_dir = self.root / "build"
        self.command_log = self.root / "commands.log"
        source_root = pathlib.Path(__file__).resolve().parents[2]
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

    def run_script(self, python_exe):
        return subprocess.run(
            ["/bin/bash", str(self.script), "-d", "CPU", "-c", "Release", "-p", str(python_exe)],
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
        source_root = pathlib.Path(__file__).resolve().parents[2]
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[0], f"-m pip install -r {source_root}/tools/ci_build/github/linux/python/requirements.txt"
        )
        self.assertTrue(calls[1].startswith(f"{source_root}/tools/ci_build/build.py --build_dir {self.build_dir} "))


if __name__ == "__main__":
    unittest.main()
