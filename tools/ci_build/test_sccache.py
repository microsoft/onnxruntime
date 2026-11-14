# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

build = importlib.import_module("build")
benchmark_sccache = importlib.import_module("benchmark_windows_sccache")
diagnose_sccache = importlib.import_module("diagnose_windows_sccache")
_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("sccache_policy", _REPO / ".github/actions/setup-sccache/cache.py")
cache = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cache)


class SccachePolicyTest(unittest.TestCase):
    def test_diagnostics_extract_actual_preprocessing_arguments_without_tracking_normalization(self):
        self.assertEqual(
            diagnose_sccache.preprocessing_commands(
                "debug unrelated command\n"
                r'preprocess: Some("C:\\Tools\\cl.exe" "-EP" "/DVERSION=\"1\"" "C:\\Source.cpp")'
            ),
            [[r"C:\Tools\cl.exe", "-EP", '/DVERSION="1"', r"C:\Source.cpp"]],
        )

    def test_diagnostics_pairs_tracking_commands_with_source_and_rejects_missing_source(self):
        self.assertEqual(
            list(diagnose_sccache.command_records('^C:\\SOURCE.CPP\n/c /Z7 "C:\\SOURCE.CPP"\n')),
            [("C:\\SOURCE.CPP", '/c /Z7 "C:\\SOURCE.CPP"')],
        )
        with self.assertRaisesRegex(ValueError, "without a source"):
            list(diagnose_sccache.command_records("/c source.cpp"))

    @staticmethod
    def e2e_script(step):
        workflow = (_REPO / ".github/workflows/windows_x64_release_build_x64_release.yml").read_text(encoding="utf-8")
        return textwrap.dedent(
            workflow.split(f"      - name: {step}\n", 1)[1]
            .split("        run: |\n", 1)[1]
            .split("\n      - name:", 1)[0]
        )

    def test_e2e_experiment_rejects_main_tags_unsafe_ids_and_extra_benchmark(self):
        script = self.e2e_script("Validate E2E cache experiment")
        for ref, experiment, benchmark, valid in (
            ("refs/heads/feature", "trial-1", "false", True),
            ("refs/heads/main", "trial-1", "false", False),
            ("refs/tags/trial", "trial-1", "false", False),
            ("refs/heads/feature", "", "false", False),
            ("refs/heads/feature", "../shared", "false", False),
            ("refs/heads/feature", "trial-1", "true", False),
        ):
            with (
                self.subTest(ref=ref, experiment=experiment, benchmark=benchmark),
                mock.patch.dict(
                    os.environ, {"GITHUB_REF": ref, "EXPERIMENT_ID": experiment, "TARGET_BENCHMARK": benchmark}
                ),
            ):
                if valid:
                    exec(script, {})
                else:
                    with self.assertRaises(ValueError):
                        exec(script, {})

    def test_e2e_experiment_uses_empty_isolated_seed_and_readonly_warm_without_spoofing_main(self):
        script = self.e2e_script("Configure E2E cache experiment")
        for mode in ("uncached", "seed", "warm"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                environment = {
                    "EXPERIMENT_MODE": mode,
                    "EXPERIMENT_ID": "trial-1",
                    "RUNNER_TEMP": directory,
                    "GITHUB_RUN_ID": "42",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_ENV": str(root / "env"),
                    "GITHUB_OUTPUT": str(root / "outputs"),
                    "CACHE_REF": "refs/heads/feature",
                }
                detect = mock.Mock()
                configure = mock.Mock()
                with (
                    mock.patch.dict(os.environ, environment),
                    mock.patch(
                        "runpy.run_path",
                        return_value={"append_values": cache.append_values, "detect": detect, "configure": configure},
                    ),
                ):
                    exec(script, {})
                    self.assertEqual(os.environ["CACHE_REF"], "refs/heads/feature")
                    exported = (root / "env").read_text()
                    flags = (root / "outputs").read_text()
                    if mode == "uncached":
                        detect.assert_not_called()
                        configure.assert_not_called()
                        self.assertEqual(exported, "SCCACHE_ENABLED=false\n")
                        self.assertEqual(flags, "build-flags=\n")
                    else:
                        detect.assert_called_once()
                        configure.assert_called_once()
                        self.assertIn(f"SCCACHE_DIR={root / 'sccache-e2e-trial-1'}", exported)
                        self.assertIn("READ_WRITE" if mode == "seed" else "READ_ONLY", exported)
                        self.assertEqual(list((root / "sccache-e2e-trial-1").iterdir()), [])
                        self.assertIn("build-flags=--use_cache --cache_tool sccache", flags)
                        with self.assertRaises(FileExistsError):
                            exec(script, {})

    def test_e2e_evidence_requires_complete_same_commit_reuse_and_readonly_warm_mode(self):
        script = self.e2e_script("Record E2E compiler-cache evidence")
        for mode, hits, misses, access, valid in (
            ("uncached", 0, 0, "", True),
            ("seed", 0, 3, "ReadWrite", True),
            ("warm", 738, 0, "ReadOnly", True),
            ("warm", 0, 0, "ReadOnly", False),
            ("warm", 1, 737, "ReadOnly", False),
            ("warm", 738, 1, "ReadOnly", False),
            ("warm", 738, 0, "ReadWrite", False),
        ):
            with (
                self.subTest(mode=mode, hits=hits, misses=misses, access=access),
                tempfile.TemporaryDirectory() as directory,
            ):
                stats = {
                    "effective_cache_mode": access,
                    "stats": {
                        "cache_hits": {"counts": {"C/C++": hits}},
                        "cache_misses": {"counts": {"C/C++": misses}},
                    },
                }
                environment = {
                    "EXPERIMENT_MODE": mode,
                    "EXPERIMENT_ID": "trial-1",
                    "GITHUB_SHA": "commit",
                    "GITHUB_RUN_ID": "42",
                    "RUNNER_NAME": "test-runner",
                    "RUNNER_TEMP": directory,
                }
                with (
                    mock.patch.dict(os.environ, environment),
                    mock.patch.object(subprocess, "check_output", return_value=json.dumps(stats)) as output,
                ):
                    if valid:
                        exec(script, {})
                    else:
                        with self.assertRaises(RuntimeError):
                            exec(script, {})
                    self.assertEqual(output.call_count, int(mode != "uncached"))
                    result = json.loads((Path(directory) / "sccache-e2e-results.json").read_text())
                    self.assertEqual(result["mode"], mode)
                    self.assertEqual(result["commit"], "commit")

    @unittest.skipUnless(
        os.name != "nt" and shutil.which("bash"), "requires a POSIX host with Bash for the macOS workflow"
    )
    def test_macos_commands_keep_cache_flags_with_newline_terminated_build_flags(self):
        workflow = (_REPO / ".github/workflows/macos-ci-build-and-test-workflow.yml").read_text(encoding="utf-8")
        commands = [line.strip() for line in workflow.splitlines() if "python ./tools/ci_build/build.py --" in line]
        self.assertEqual(len(commands), 3)
        for flags in ("--use_cache --cache_tool sccache", ""):
            for command in commands:
                with self.subTest(flags=flags, command=command):
                    expanded = command.replace("${{ steps.sccache.outputs.build-flags }}", flags).replace(
                        "${{ env.build_flags }}", "--config Debug\n"
                    )
                    result = subprocess.run(
                        ["bash", "-c", 'python() { printf "%s\\n" "$@"; }\n' + expanded],
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = ["./tools/ci_build/build.py", command.split()[2]]
                    if "sccache.outputs" in command:
                        expected.extend(flags.split())
                    self.assertEqual(result.stdout.splitlines(), [*expected, "--config", "Debug"])

    def test_selects_exact_or_compatible_snapshot_and_populates_only_main(self):
        for ref, matched, expected in (
            ("refs/heads/main", "", True),
            ("refs/heads/main", "exact", True),
            ("refs/pull/42/merge", "", False),
            ("refs/pull/42/merge", "exact", True),
            ("refs/pull/42/merge", "compatible-prefix-older-sha", True),
            ("refs/heads/rel-1.25", "", False),
            ("refs/heads/not-main", "", False),
        ):
            with self.subTest(ref=ref, matched=matched), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                environment = {
                    "CACHE_REF": ref,
                    "CACHE_MATCHED_KEY": matched,
                    "SCCACHE_LOCAL_RW_MODE": "READ_WRITE" if ref == "refs/heads/main" else "READ_ONLY",
                    "GITHUB_ENV": str(root / "env"),
                    "GITHUB_OUTPUT": str(root / "outputs"),
                    "GITHUB_STEP_SUMMARY": str(root / "summary"),
                }
                with mock.patch.dict(os.environ, environment):
                    cache.select()
                self.assertIn(f"SCCACHE_ENABLED={str(expected).lower()}", (root / "env").read_text())
                self.assertIn(
                    "build-flags=" + ("--use_cache --cache_tool sccache\n" if expected else "\n"),
                    (root / "outputs").read_text(),
                )

    def test_windows_requires_preinstalled_cache_in_selected_compiler_environment(self):
        with (
            mock.patch.dict(os.environ, {"RUNNER_OS": "Windows"}),
            mock.patch.object(cache.shutil, "which", return_value=None),
            self.assertRaisesRegex(FileNotFoundError, "preinstalled"),
        ):
            cache.detect()

    def test_windows_identity_tracks_compiler_sdk_and_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / "cl.exe"
            compiler.write_bytes(b"compiler version one")
            environment = {
                "RUNNER_OS": "Windows",
                "RUNNER_ARCH": "X64",
                "CACHE_VARIANT": "cuda-release",
                "CACHE_SIZE": "10G",
                "CACHE_REF": "refs/heads/main",
                "GITHUB_WORKSPACE": directory,
                "GITHUB_ENV": str(root / "env"),
                "GITHUB_OUTPUT": str(root / "outputs"),
                "WindowsSDKVersion": "10.0.26100.0",
            }

            def which(name):
                return str(compiler) if name == "cl" else name

            with (
                mock.patch.dict(os.environ, environment),
                mock.patch.object(cache.shutil, "which", side_effect=which),
                mock.patch.object(cache.subprocess, "check_output") as version,
            ):
                version.side_effect = lambda command, **_: "sccache 0.19.0" if command[0] == "sccache" else "CUDA 13.0"
                cache.configure()
                first = (root / "outputs").read_text().splitlines()[-1]
                cache.configure()
                self.assertEqual(first, (root / "outputs").read_text().splitlines()[-1])
                compiler.write_bytes(b"compiler version two")
                cache.configure()
                second = (root / "outputs").read_text().splitlines()[-1]
                self.assertNotEqual(first, second)
                os.environ["WindowsSDKVersion"] = "10.0.28000.0"  # noqa: SIM112
                cache.configure()
                third = (root / "outputs").read_text().splitlines()[-1]
                self.assertNotEqual(second, third)
                version.side_effect = lambda command, **_: "sccache 0.19.0" if command[0] == "sccache" else "CUDA 13.1"
                cache.configure()
                self.assertNotEqual(third, (root / "outputs").read_text().splitlines()[-1])
            self.assertIn("SCCACHE_LOCAL_RW_MODE=READ_WRITE", (root / "env").read_text())

    @unittest.skipUnless(
        os.name == "nt" and all(shutil.which(tool) for tool in ("sccache", "cl", "cmake")),
        "requires a Windows developer environment with preinstalled sccache and CMake",
    )
    def test_visual_studio_rebuilds_use_the_native_and_optional_cuda_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            cuda = shutil.which("nvcc")
            cuda_home = str(Path(cuda).parent.parent) if cuda else ""
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.28)\n"
                "project(cache_test LANGUAGES C CXX" + (" CUDA" if cuda else "") + ")\n"
                "add_executable(cache_test main.cpp value.c" + (" kernel.cu" if cuda else "") + ")\n",
                encoding="utf-8",
            )
            (source / "main.cpp").write_text(
                'extern "C" int value(void);\nint main() { return value() == 42 ? 0 : 1; }\n', encoding="utf-8"
            )
            header = source / "value.h"
            header.write_text("#define VALUE 42\n", encoding="utf-8")
            (source / "value.c").write_text('#include "value.h"\nint value(void) { return VALUE; }\n', encoding="utf-8")
            if cuda:
                (source / "kernel.cu").write_text("__global__ void unused_kernel() {}\n", encoding="utf-8")
            args = SimpleNamespace(
                cache_tool="sccache",
                cmake_generator="Visual Studio 17 2022",
                arm64=False,
                arm64ec=False,
                arm=False,
                x86=False,
                msvc_toolset=None,
                use_cuda=bool(cuda),
            )
            build_dir = root / "build"
            globals_ = build.windows_sccache_globals(args, str(build_dir), cuda_home)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            environment = dict(
                os.environ,
                SCCACHE_DIR=str(root / "cache"),
                SCCACHE_SERVER_PORT=str(port),
                SCCACHE_LOCAL_RW_MODE="READ_WRITE",
            )

            def run(*command):
                result = subprocess.run(
                    command, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
                )
                self.assertEqual(result.returncode, 0, result.stdout)
                return result.stdout

            run("sccache", "--start-server")
            try:
                run(
                    "cmake",
                    "-S",
                    str(source),
                    "-B",
                    str(build_dir),
                    "-G",
                    args.cmake_generator,
                    "-DCMAKE_VS_GLOBALS="
                    + ";".join([*globals_, "UseMultiToolTask=true", "EnforceProcessCountAcrossBuilds=true"]),
                    "-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded",
                )
                run("sccache", "--zero-stats")
                for iteration in range(2):
                    run("cmake", "--build", str(build_dir), "--config", "RelWithDebInfo", "--clean-first")
                    run(str(build_dir / "RelWithDebInfo" / "cache_test.exe"))
                    stats = json.loads(run("sccache", "--show-stats", "--stats-format=json"))["stats"]
                    self.assertEqual(stats["compile_fails"], 0)
                    self.assertEqual(stats["requests_not_cacheable"], 0, stats)
                    counts = stats["cache_misses" if iteration == 0 else "cache_hits"]["counts"]
                    self.assertGreaterEqual(counts.get("C/C++", 0), 2, stats)
                    if cuda:
                        self.assertGreaterEqual(counts.get("CUDA", 0), 1, stats)
                # Without FileTracker, header-only edits must still reach the compiler cache.
                header.write_text("#define VALUE 43\n", encoding="utf-8")
                run("sccache", "--zero-stats")
                run("cmake", "--build", str(build_dir), "--config", "RelWithDebInfo")
                result = subprocess.run([str(build_dir / "RelWithDebInfo" / "cache_test.exe")], check=False)
                self.assertEqual(result.returncode, 1)
                stats = json.loads(run("sccache", "--show-stats", "--stats-format=json"))["stats"]
                self.assertGreaterEqual(stats["cache_misses"]["counts"].get("C/C++", 0), 1, stats)
                header.write_text("#define VALUE 42\n", encoding="utf-8")
                run("sccache", "--zero-stats")
                run("cmake", "--build", str(build_dir), "--config", "RelWithDebInfo")
                run(str(build_dir / "RelWithDebInfo" / "cache_test.exe"))
                stats = json.loads(run("sccache", "--show-stats", "--stats-format=json"))["stats"]
                self.assertGreaterEqual(stats["cache_hits"]["counts"].get("C/C++", 0), 2, stats)
            finally:
                run("sccache", "--stop-server")

    def test_benchmark_cleans_outputs_and_requires_cold_and_warm_cache_evidence(self):
        for warm_hits in (2, 0):
            with self.subTest(warm_hits=warm_hits), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "CMakeCache.txt").write_text(
                    f"CMAKE_HOME_DIRECTORY:INTERNAL={root}\nCMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022\n"
                    "CMAKE_GENERATOR_PLATFORM:INTERNAL=x64\n",
                    encoding="utf-8",
                )
                calls = []

                def run(command, calls=calls, **kwargs):
                    calls.append((command, kwargs["env"].copy()))

                def output(command, warm_hits=warm_hits, **kwargs):
                    if command == ["sccache", "--version"]:
                        return "sccache 0.19.0\n"
                    if command == ["where.exe", "cl"]:
                        return "cl.exe\n"
                    warm = kwargs["env"]["SCCACHE_LOCAL_RW_MODE"] == "READ_ONLY"
                    return json.dumps(
                        {
                            "cache_location": "Local disk: isolated benchmark cache",
                            "stats": {
                                "compile_fails": 0,
                                "cache_misses": {"counts": {"C/C++": 2 if not warm else 0}},
                                "cache_hits": {"counts": {"C/C++": warm_hits if warm else 0}},
                            },
                        }
                    )

                with (
                    mock.patch.object(benchmark_sccache.platform, "platform", return_value="test-platform"),
                    mock.patch.object(benchmark_sccache, "windows_sccache_globals", return_value=["CLToolExe=cl.exe"]),
                    mock.patch.object(benchmark_sccache.subprocess, "run", side_effect=run),
                    mock.patch.object(benchmark_sccache.subprocess, "check_output", side_effect=output),
                    mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}),
                ):
                    if warm_hits:
                        result = benchmark_sccache.benchmark(root, "RelWithDebInfo", "sample", 2, root / "results")
                        self.assertIn("warm_speedup", result)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "No successful warm cache evidence"):
                            benchmark_sccache.benchmark(root, "RelWithDebInfo", "sample", 2, root / "results")
                builds = [(cmd, env) for cmd, env in calls if cmd[:2] == ["cmake", "--build"]]
                self.assertEqual(len(builds), 3)
                self.assertTrue(all("--clean-first" in cmd for cmd, _ in builds))
                self.assertEqual(builds[-1][1]["SCCACHE_LOCAL_RW_MODE"], "READ_ONLY")
                self.assertNotEqual(builds[-1][1]["SCCACHE_DIR"], os.environ.get("SCCACHE_DIR"))
                stops = [cmd for cmd, _ in calls if cmd == ["sccache", "--stop-server"]]
                self.assertEqual(len(stops), 2)


if __name__ == "__main__":
    unittest.main()
