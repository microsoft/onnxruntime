# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache = load_module("cache")


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {
            "CACHE_VARIANT": "linux-cpu-x64-Release",
            "CACHE_SIZE": "10G",
            "CACHE_REF": "refs/heads/main",
            "CACHE_PREFIX": "sccache-linux-v1",
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
            "GITHUB_WORKSPACE": str(self.root),
            "RUNNER_TEMP": str(self.root),
            "GITHUB_ENV": str(self.root / "env"),
            "GITHUB_OUTPUT": str(self.root / "output"),
            "GITHUB_STEP_SUMMARY": str(self.root / "summary"),
            "SCCACHE_CACHE_SIZE": "10G",
            "SCCACHE_LOCAL_RW_MODE": "READ_ONLY",
        }

    def values(self, name):
        return dict(line.split("=", 1) for line in (self.root / name).read_text(encoding="utf-8").splitlines())

    def test_exact_fallback_and_miss_on_main_and_nonpublishing_refs(self):
        for ref in ("refs/heads/main", "refs/pull/42/merge", "refs/heads/feature", "refs/heads/rel-1"):
            for matched in ("exact-primary-key", "compatible-older-sha", ""):
                with self.subTest(ref=ref, matched=matched):
                    self.env.update(CACHE_REF=ref, CACHE_MATCHED_KEY=matched, SCCACHE_CONTAINER_IMAGE="image")
                    with patch.dict(os.environ, self.env, clear=True):
                        cache.select()
                    enabled = ref == "refs/heads/main" or bool(matched)
                    self.assertEqual(self.values("output")["enabled"], str(enabled).lower())
                    self.assertEqual(
                        self.values("output")["build-flags"], "--use_cache --cache_tool sccache" if enabled else ""
                    )
                    self.assertIn(
                        f"export SCCACHE_ENABLED={str(enabled).lower()}", (self.root / "ort-sccache.env").read_text()
                    )

    def test_host_installed_binary_is_used_and_version_is_in_key(self):
        prefixes = []
        for version in ("sccache 0.19.0", "sccache 0.19.1"):
            with (
                patch.dict(os.environ, self.env, clear=True),
                patch.object(cache.shutil, "which", return_value="/preinstalled/sccache"),
                patch.object(cache.subprocess, "check_output", return_value=version) as run,
            ):
                cache.configure()
            run.assert_called_once_with(["/preinstalled/sccache", "--version"], text=True)
            prefixes.append(self.values("output")["prefix"])
        self.assertNotEqual(*prefixes)

    def test_missing_and_invalid_host_tool_fail_explicitly(self):
        with (
            patch.dict(os.environ, self.env, clear=True),
            patch.object(cache.shutil, "which", return_value=None),
            self.assertRaises(FileNotFoundError),
        ):
            cache.configure()
        with (
            patch.dict(os.environ, self.env, clear=True),
            patch.object(cache.shutil, "which", return_value="sccache"),
            patch.object(cache.subprocess, "check_output", return_value=""),
            self.assertRaisesRegex(ValueError, "Invalid sccache"),
        ):
            cache.configure()

    def test_container_probe_versions_image_and_toolchain_partition_keys(self):
        self.env.update(CACHE_CONTAINER_IMAGE="build-image", CACHE_CONTAINER_PREFIX="PATH=/opt/python/cp311/bin:$PATH")
        prefixes = []
        for image, compiler, version in (
            ("sha256:a", "gcc 14.2", "0.19.0"),
            ("sha256:b", "gcc 14.2", "0.19.0"),
            ("sha256:b", "gcc 14.3", "0.19.0"),
            ("sha256:b", "gcc 14.3", "0.19.1"),
        ):
            with (
                patch.dict(os.environ, self.env, clear=True),
                patch.object(
                    cache.subprocess,
                    "check_output",
                    side_effect=[f"/usr/bin/sccache\nsccache {version}\n{compiler}", image],
                ) as run,
            ):
                cache.configure()
            command = run.call_args_list[0].args[0]
            self.assertEqual(command[:5], ["docker", "run", "--rm", "build-image", "/bin/bash"])
            self.assertIn("PATH=/opt/python/cp311/bin:$PATH", command[-1])
            self.assertIn("sccache --version", command[-1])
            prefixes.append(self.values("output")["prefix"])
        self.assertEqual(len(set(prefixes)), 4)

    def test_missing_container_tool_or_failed_probe_does_not_install_on_host(self):
        self.env["CACHE_CONTAINER_IMAGE"] = "build-image"
        for result in ("gcc 14.2", "sccache \nsccache \n"):
            with (
                patch.dict(os.environ, self.env, clear=True),
                patch.object(cache.subprocess, "check_output", side_effect=[result, "sha256:a"]),
                self.assertRaisesRegex(ValueError, "installed sccache version"),
            ):
                cache.configure()
        with (
            patch.dict(os.environ, self.env, clear=True),
            patch.object(cache.subprocess, "check_output", side_effect=subprocess.CalledProcessError(127, ["docker"])),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            cache.configure()

    def test_main_writes_and_other_refs_are_read_only(self):
        for ref, mode in (("refs/heads/main", "READ_WRITE"), ("refs/pull/42/merge", "READ_ONLY")):
            self.env["CACHE_REF"] = ref
            with (
                patch.dict(os.environ, self.env, clear=True),
                patch.object(cache.shutil, "which", return_value="sccache"),
                patch.object(cache.subprocess, "check_output", return_value="sccache 0.19.0"),
            ):
                cache.configure()
            self.assertEqual(self.values("env")["SCCACHE_LOCAL_RW_MODE"], mode)

    def test_verifies_generated_launchers_in_actual_build_directory(self):
        build_dir = self.root / "Release" / "Release"
        build_dir.mkdir(parents=True)
        cmake_cache = build_dir / "CMakeCache.txt"
        for enabled, launcher in (("true", "sccache"), ("false", "")):
            for use_cuda in (False, True):
                with self.subTest(enabled=enabled, use_cuda=use_cuda):
                    self.env.update(BUILD_CONFIG="Release", SCCACHE_ENABLED=enabled)
                    contents = (
                        "CMAKE_C_COMPILER:FILEPATH=/usr/bin/cc\nCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n"
                        f"CMAKE_C_COMPILER_LAUNCHER:UNINITIALIZED={launcher}\n"
                        f"CMAKE_CXX_COMPILER_LAUNCHER:UNINITIALIZED={launcher}\n"
                    )
                    if use_cuda:
                        contents += (
                            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda-13.0/bin/nvcc\n"
                            f"CMAKE_CUDA_COMPILER_LAUNCHER:STRING={launcher}\n"
                        )
                    cmake_cache.write_text(contents, encoding="utf-8")
                    with patch.dict(os.environ, self.env, clear=True):
                        cache.verify_launchers()
                        with cmake_cache.open("a", encoding="utf-8") as output:
                            output.write("CMAKE_CUDA_COMPILER_LAUNCHER:STRING=ccache\n")
                        with self.assertRaisesRegex(ValueError, "CUDA"):
                            cache.verify_launchers()
        cmake_cache.write_text("CMAKE_C_COMPILER_LAUNCHER:STRING=ccache\n", encoding="utf-8")
        with patch.dict(os.environ, self.env, clear=True), self.assertRaisesRegex(ValueError, "expected"):
            cache.verify_launchers()

    def test_missing_cuda_launcher_is_rejected_when_caching_is_enabled(self):
        build_dir = self.root / "Debug" / "Debug"
        build_dir.mkdir(parents=True)
        (build_dir / "CMakeCache.txt").write_text(
            "CMAKE_C_COMPILER:FILEPATH=/usr/bin/cc\nCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n"
            "CMAKE_C_COMPILER_LAUNCHER:STRING=sccache\nCMAKE_CXX_COMPILER_LAUNCHER:STRING=sccache\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda-12.8/bin/nvcc\n",
            encoding="utf-8",
        )
        self.env.update(BUILD_CONFIG="Debug", SCCACHE_ENABLED="true")
        with patch.dict(os.environ, self.env, clear=True), self.assertRaisesRegex(ValueError, "CMAKE_CUDA"):
            cache.verify_launchers()

    def test_container_report_requires_flushed_server_and_never_uses_host_tool(self):
        self.env["SCCACHE_CONTAINER_IMAGE"] = "build-image"
        with patch.dict(os.environ, self.env, clear=True), patch.object(cache.subprocess, "check_output") as run:
            with self.assertRaisesRegex(RuntimeError, "stopped/flushed"):
                cache.report()
            (self.root / "sccache-flushed").touch()
            (self.root / "sccache-stats.txt").write_text("Cache hits 42\n", encoding="utf-8")
            cache.report()
            run.assert_not_called()
        self.assertIn("Cache hits 42", (self.root / "summary").read_text())

    @unittest.skipUnless(shutil.which("bash"), "bash is required for the image installer test")
    def test_image_installer_rejects_unsupported_arch_and_failed_or_corrupt_downloads(self):
        script = Path(__file__).resolve().parents[3] / "tools/ci_build/github/linux/docker/scripts/install_sccache.sh"
        (self.root / "bin").mkdir()
        (self.root / "tmp").mkdir()
        for name, contents in {
            "uname": '#!/bin/bash\necho "$TEST_ARCH"\n',
            "curl": (
                '#!/bin/bash\necho download >> "$PWD/commands"\n'
                'if [[ "$FAIL_DOWNLOAD" == true ]]; then echo "Download failed" >&2; exit 22; fi\n'
                'while [[ "$1" != "--output" ]]; do shift; done\n'
                'echo corrupt-download > "$2"\n'
            ),
        }.items():
            tool = self.root / "bin" / name
            tool.write_text(contents, encoding="utf-8", newline="\n")
            tool.chmod(0o755)
        for arch, fail_download, expected, message in (
            ("unsupported", "false", 1, "Unsupported sccache architecture"),
            ("x86_64", "true", 22, "Download failed"),
            ("x86_64", "false", 1, "FAILED"),
            ("aarch64", "false", 1, "FAILED"),
        ):
            with self.subTest(arch=arch, fail_download=fail_download):
                (self.root / "commands").unlink(missing_ok=True)
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'export PATH="$PWD/bin:$PATH" TMPDIR="$PWD/tmp"; bash "$1" "$PWD/installed"',
                        "test",
                        str(script),
                    ],
                    cwd=self.root,
                    env={**os.environ, "TEST_ARCH": arch, "FAIL_DOWNLOAD": fail_download},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                self.assertIn(message, result.stdout + result.stderr)
                self.assertEqual((self.root / "commands").exists(), arch != "unsupported")
                self.assertFalse((self.root / "installed").exists())
                self.assertEqual(list((self.root / "tmp").iterdir()), [])

    @unittest.skipUnless(
        sys.platform.startswith("linux") and all(shutil.which(tool) for tool in ("cmake", "ninja", "sccache", "g++")),
        "Linux CMake, Ninja, sccache, and GCC are required for the PCH regression test",
    )
    def test_gcc_plugin_pch_bypasses_cache_and_header_changes_take_effect(self):
        self.check_plugin_pch("g++", cached_pch=False)

    @unittest.skipUnless(
        sys.platform.startswith("linux")
        and all(shutil.which(tool) for tool in ("cmake", "ninja", "sccache", "clang++")),
        "Linux CMake, Ninja, sccache, and Clang are required for the PCH regression test",
    )
    def test_clang_plugin_pch_is_cached_and_header_changes_take_effect(self):
        self.check_plugin_pch("clang++", cached_pch=True)

    def check_plugin_pch(self, compiler, cached_pch):
        cmake = (Path(__file__).resolve().parents[3] / "cmake/onnxruntime_providers_webgpu.cmake").read_text()
        # Exercise the production guard without configuring all ORT/Dawn dependencies.
        start = cmake.index('    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_LAUNCHER')
        end = cmake.index("    endif()", start) + len("    endif()")
        (self.root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\nproject(pch_test LANGUAGES CXX)\n"
            "add_executable(onnxruntime_providers_webgpu main.cpp)\n"
            'target_precompile_headers(onnxruntime_providers_webgpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/pch.h")\n'
            + cmake[start:end]
            + "\nadd_library(ordinary STATIC ordinary.cpp)\n",
            encoding="utf-8",
        )
        header = self.root / "pch.h"
        header.write_text("#include <vector>\n#define PCH_VALUE 1\n", encoding="utf-8")
        main = self.root / "main.cpp"
        main.write_text("int main() { std::vector<int> values{PCH_VALUE}; return values[0]; }\n", encoding="utf-8")
        (self.root / "ordinary.cpp").write_text("int ordinary() { return 7; }\n", encoding="utf-8")
        env = {
            **os.environ,
            "SCCACHE_DIR": str(self.root / "cache"),
            "SCCACHE_LOCAL_RW_MODE": "READ_WRITE",
            "SCCACHE_SERVER_UDS": str(self.root / "server.sock"),
        }

        def run(*command):
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

        build = self.root / "build"
        run(
            "cmake",
            "-S",
            str(self.root),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_CXX_COMPILER={shutil.which(compiler)}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={shutil.which('sccache')}",
        )
        run("sccache", "--start-server")
        try:
            for index, expected in enumerate((1, 1, 2, 3) if cached_pch else (1, 1, 2)):
                if expected == 2:
                    header.write_text("#include <vector>\n#define PCH_VALUE 2\n", encoding="utf-8")
                elif expected == 3:
                    # A new checkout can have unchanged headers with different timestamps.
                    timestamp = header.stat().st_mtime + 2
                    os.utime(header, (timestamp, timestamp))
                    main.write_text(
                        "int main() { std::vector<int> values{PCH_VALUE}; return values[0] + 1; }\n",
                        encoding="utf-8",
                    )
                run("sccache", "--zero-stats")
                run("cmake", "--build", str(build), "--clean-first")
                result = subprocess.run([str(build / "onnxruntime_providers_webgpu")], check=False)
                self.assertEqual(result.returncode, expected)
                stats = json.loads(run("sccache", "--show-stats", "--stats-format", "json"))["stats"]
                requests = 3 if cached_pch else 1
                self.assertEqual(stats["compile_requests"], requests)
                if index == 1:
                    self.assertEqual(sum(stats["cache_hits"]["counts"].values()), requests)
        finally:
            run("sccache", "--stop-server")

    @unittest.skipUnless(shutil.which("bash"), "bash is required for the container lifecycle test")
    def test_ephemeral_container_lifecycle_and_failures(self):
        script = Path(__file__).with_name("container.sh").resolve()
        (self.root / "bin").mkdir()
        (self.root / "build").mkdir()
        (self.root / ".sccache").mkdir()
        tool = self.root / "bin" / "sccache"
        tool.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$*" >> build/commands\n'
            'if [[ "$1" == "--show-stats" ]]; then echo "Cache hits 42"; fi\n'
            'if [[ "$1" == "$FAIL_COMMAND" ]]; then exit 7; fi\n',
            encoding="utf-8",
            newline="\n",
        )
        tool.chmod(0o755)
        # Exercise the same && context as the pinned Docker action, where errexit alone is insufficient.
        command = (
            'set -ex; export PATH="$PWD/bin:$PATH"; source "$1" && '
            'echo compiled > build/compiled && (exit "$COMPILE_STATUS")'
        )
        for enabled, fail, compile_status, expected in (
            ("true", "", "0", 0),
            ("true", "", "3", 3),
            ("true", "--start-server", "0", 7),
            ("true", "--zero-stats", "0", 7),
            ("true", "--show-stats", "0", 1),
            ("true", "--stop-server", "0", 1),
            ("false", "", "0", 0),
        ):
            with self.subTest(enabled=enabled, fail=fail, compile_status=compile_status):
                for name in ("commands", "sccache-flushed", "compiled"):
                    (self.root / "build" / name).unlink(missing_ok=True)
                (self.root / "build" / "ort-sccache.env").write_text(
                    f"export SCCACHE_ENABLED={enabled}\n"
                    'export SCCACHE_DIR="$PWD/.sccache"\nexport SCCACHE_LOCAL_RW_MODE=READ_WRITE\n',
                    encoding="utf-8",
                    newline="\n",
                )
                result = subprocess.run(
                    ["bash", "-c", command, "test", str(script)],
                    check=False,
                    cwd=self.root,
                    env={**os.environ, "FAIL_COMMAND": fail, "COMPILE_STATUS": compile_status},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                commands = self.root / "build" / "commands"
                if enabled == "false":
                    self.assertFalse(commands.exists())
                else:
                    calls = commands.read_text().splitlines()
                    self.assertIn("--stop-server", calls)
                    self.assertEqual(calls[-2:], ["--show-stats", "--stop-server"])
                    if fail in ("--start-server", "--zero-stats"):
                        self.assertFalse((self.root / "build" / "compiled").exists())
                    self.assertEqual(
                        (self.root / "build" / "sccache-flushed").exists(),
                        fail not in ("--show-stats", "--stop-server"),
                    )


if __name__ == "__main__":
    unittest.main()
