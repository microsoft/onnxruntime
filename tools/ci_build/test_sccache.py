# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

build = importlib.import_module("build")
_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("sccache_policy", _REPO / ".github/actions/setup-sccache/cache.py")
cache = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cache)


class SccachePolicyTest(unittest.TestCase):
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
            (source / "value.c").write_text("int value(void) { return 42; }\n", encoding="utf-8")
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
            finally:
                run("sccache", "--stop-server")


if __name__ == "__main__":
    unittest.main()
