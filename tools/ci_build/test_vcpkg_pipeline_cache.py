# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Validate ADO vcpkg cache wiring without building dependencies or running Docker."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

_CI_BUILD = Path(__file__).resolve().parent
_PIPELINES = _CI_BUILD / "github" / "azure-pipelines"
_TEMPLATE = _PIPELINES / "templates" / "setup-vcpkg-cache.yml"
_DOCKER_WRAPPERS = (
    "run_python_dockerbuild.sh",
    "build_cuda_c_api_package.sh",
    "build_tensorrt_c_api_package.sh",
    "build_cuda_plugin_package.sh",
    "build_webgpu_plugin_package.sh",
    "build_nodejs_package.sh",
)


class VcpkgPipelineCacheTest(unittest.TestCase):
    def test_cache_keys_track_inputs_and_restore_across_commits(self):
        template = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
        setup, cache = template["steps"]
        self.assertEqual(cache["task"], "Cache@2")
        self.assertEqual(setup["env"]["VCPKG_CACHE_PATH"], cache["inputs"]["path"])
        key = cache["inputs"]["key"]
        restore_condition = "${{ if not(startsWith(variables['Build.SourceBranch'], 'refs/heads/rel-')) }}"
        prefix = cache["inputs"][restore_condition]["restoreKeys"].strip()
        self.assertTrue(key.startswith(prefix + " |"))
        for component in ("$(Agent.OS)", "$(Agent.OSArchitecture)", "$(System.DefinitionId)", "${{ parameters.key }}"):
            self.assertIn(component, prefix)
        self.assertIn("$(Build.SourceVersion)", key)
        self.assertNotIn("$(Build.SourceVersion)", prefix)
        for pattern in key.split("|")[-2].strip().split(","):
            self.assertTrue(list(_CI_BUILD.parents[1].glob(pattern)), pattern)
        self.assertNotIn("condition", cache)

    def test_release_branches_have_no_fallback_restore_keys(self):
        inputs = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))["steps"][1]["inputs"]
        restore_condition = "${{ if not(startsWith(variables['Build.SourceBranch'], 'refs/heads/rel-')) }}"
        self.assertNotIn("restoreKeys", inputs)
        self.assertEqual(
            [name for name, value in inputs.items() if isinstance(value, dict) and "restoreKeys" in value],
            [restore_condition],
        )
        self.assertIn("key", inputs)
        self.assertIn("path", inputs)

    def test_linux_gpu_jobs_use_distinct_keys_under_1es(self):
        stage = yaml.safe_load(
            (_PIPELINES / "stages" / "nuget-linux-cuda-packaging-stage.yml").read_text(encoding="utf-8")
        )["stages"][0]
        keys = []
        for job in stage["jobs"]:
            cache_steps = [step for step in job["steps"] if step.get("template", "").endswith("setup-vcpkg-cache.yml")]
            self.assertEqual(len(cache_steps), 1)
            key = cache_steps[0]["parameters"]["key"]
            self.assertEqual(key, f"$(System.StageName)-{job['job']}")
            self.assertNotIn("$(System.JobName)", key)
            keys.append(key)
        self.assertEqual(len(keys), len(set(keys)))

    def test_windows_nodejs_builds_use_vcpkg_with_target_specific_caches(self):
        document = yaml.safe_load(
            (_PIPELINES / "stages" / "nodejs-win-packaging-stage.yml").read_text(encoding="utf-8")
        )
        steps = document["stages"][0]["jobs"][0]["steps"]
        cache = next(step for step in steps if step.get("template", "").endswith("setup-vcpkg-cache.yml"))
        key = cache["parameters"]["key"]
        self.assertIn("${{ parameters.sln_platform }}", key)
        self.assertIn("$(BuildConfig)", key)
        self.assertNotIn("$(System.JobName)", key)
        build = next(step for step in steps if "tools\\ci_build\\build.py" in step.get("powershell", ""))
        self.assertLess(steps.index(cache), steps.index(build))
        for flag in ("--use_vcpkg", "--use_vcpkg_ms_internal_asset_cache"):
            self.assertIn(flag, build["powershell"].split())
        pipeline = yaml.safe_load(
            (_PIPELINES / "c-api-noopenmp-packaging-pipelines-cuda13.yml").read_text(encoding="utf-8")
        )
        nodejs_stages = [
            stage["parameters"]
            for stage in pipeline["extends"]["parameters"]["stages"]
            if stage.get("template", "").endswith("nodejs-win-packaging-stage.yml")
        ]
        self.assertEqual({stage["sln_platform"] for stage in nodejs_stages}, {"x64", "arm64"})
        for stage in nodejs_stages:
            with self.subTest(arch=stage["sln_platform"]):
                self.assertIn("--build_nodejs", stage["BuildCommand"])
                self.assertIn("--use_dml", stage["BuildCommand"])
                self.assertIn("--use_webgpu", stage["BuildCommand"])
        arm64 = next(stage for stage in nodejs_stages if stage["sln_platform"] == "arm64")
        self.assertIn("--arm64", arm64["BuildCommand"])

    def test_vcpkg_build_steps_restore_cache_before_building(self):
        def find_steps(value):
            if isinstance(value, dict):
                if "steps" in value:
                    yield value["steps"]
                for child in value.values():
                    yield from find_steps(child)
            elif isinstance(value, list):
                for child in value:
                    yield from find_steps(child)

        def flatten_conditions(steps):
            for step in steps:
                if not isinstance(step, dict):
                    continue  # Unexpanded stepList parameters have no concrete build steps.
                for name, value in step.items():
                    if name.startswith("${{") and isinstance(value, list):
                        yield from flatten_conditions(value)
                        break
                else:
                    yield step

        builds_checked = 0
        for path in _PIPELINES.rglob("*.yml"):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            for steps in find_steps(document):
                cache_seen = False
                for step in flatten_conditions(steps):
                    if step.get("template", "").endswith("setup-vcpkg-cache.yml"):
                        cache_seen = True
                    build = json.dumps(step)
                    markers = (
                        "--use_vcpkg",
                        "build_tensorrt_ci.sh",
                        "build_aar_and_copy_artifacts.sh",
                        *_DOCKER_WRAPPERS,
                    )
                    if any(marker in build for marker in markers):
                        with self.subTest(path=path):
                            self.assertTrue(cache_seen, "vcpkg build must restore the cache first")
                            if "docker run" in build:
                                self.assertIn("VCPKG_DEFAULT_BINARY_CACHE=/vcpkg-binary-cache", build)
                                self.assertIn("${VCPKG_DEFAULT_BINARY_CACHE}:/vcpkg-binary-cache", build)
                        builds_checked += 1
        self.assertGreater(builds_checked, 0)

    @unittest.skipUnless(shutil.which("pwsh"), "Requires PowerShell Core")
    def test_setup_creates_directory_and_exports_absolute_path(self):
        setup = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))["steps"][0]
        with tempfile.TemporaryDirectory(prefix="vcpkg cache ") as directory:
            cache_path = str(Path(directory) / "binary cache")
            result = subprocess.run(
                ["pwsh", "-NoProfile", "-NonInteractive", "-Command", setup["pwsh"]],
                env={**os.environ, "VCPKG_CACHE_PATH": cache_path},
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(Path(cache_path).is_dir())
            self.assertIn(f"##vso[task.setvariable variable=VCPKG_DEFAULT_BINARY_CACHE]{cache_path}", result.stdout)

    @unittest.skipUnless(shutil.which("bash"), "Requires Bash")
    def test_docker_wrappers_mount_cache_only_when_configured(self):
        with tempfile.TemporaryDirectory(prefix="vcpkg cache ") as directory:
            for wrapper in _DOCKER_WRAPPERS:
                for configured_path in (None, "", f"{Path(directory).as_posix()}/binary cache"):
                    with self.subTest(wrapper=wrapper, configured_path=configured_path):
                        env = {
                            **os.environ,
                            "HOME": Path(directory).as_posix(),
                            "CUDA_VERSION": "12.8",
                            "SYSTEM_ACCESSTOKEN": "",
                        }
                        env.pop("VCPKG_DEFAULT_BINARY_CACHE", None)
                        if configured_path is not None:
                            env["VCPKG_DEFAULT_BINARY_CACHE"] = configured_path
                        result = subprocess.run(
                            [
                                "bash",
                                "-c",
                                # Stop at docker so no compilation, cleanup, or artifact publication runs.
                                'docker() { printf "%s\\0" "$@"; return 73; }; source "$1"',
                                "test",
                                (_CI_BUILD / "github" / "linux" / wrapper).as_posix(),
                            ],
                            env=env,
                            capture_output=True,
                            check=False,
                            timeout=30,
                        )
                        self.assertEqual(result.returncode, 73, result.stderr.decode(errors="replace"))
                        args = result.stdout.decode().split("\0")
                        mount = f"{configured_path}:/vcpkg-binary-cache"
                        setting = "VCPKG_DEFAULT_BINARY_CACHE=/vcpkg-binary-cache"
                        if configured_path:
                            self.assertIn(mount, args)
                            self.assertEqual(args[args.index(mount) - 1], "--volume")
                            self.assertIn(setting, args)
                            self.assertEqual(args[args.index(setting) - 1], "-e")
                        else:
                            self.assertNotIn(setting, args)
                            self.assertFalse(any(":/vcpkg-binary-cache" in arg for arg in args))


if __name__ == "__main__":
    unittest.main()
