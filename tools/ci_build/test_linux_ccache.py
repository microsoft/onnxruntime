# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTION = yaml.safe_load((ROOT / ".github/actions/setup-linux-ccache/action.yml").read_text())
CONFIGURE, RESTORE, ACCESS = ACTION["runs"]["steps"]
WORKFLOWS = ("reusable_linux_build.yml", "linux_minimal_build.yml")


class TestLinuxCcache(unittest.TestCase):
    def configure(self, workspace, *, variant="debug", image="image:tag", toolchain="sha256:one", **overrides):
        output = workspace / "output"
        environment = workspace / "env"
        output.write_text("")
        environment.write_text("")
        env = {
            "CACHE_VARIANT": variant,
            "DOCKER_IMAGE": image,
            "RUNNER_ARCH": "X64",
            "GITHUB_RUN_ID": "100",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_WORKSPACE": str(workspace),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_ENV": str(environment),
            **overrides,
        }
        command = patch("subprocess.check_output", return_value=toolchain) if toolchain is not None else nullcontext()
        with patch.dict(os.environ, env), command:
            exec(compile(CONFIGURE["run"], "configure-ccache", "exec"), {})
        return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_rolling_snapshots_restore_only_compatible_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            cold = self.configure(workspace)
            for overrides in ({"GITHUB_RUN_ID": "101"}, {"GITHUB_RUN_ATTEMPT": "2"}):
                with self.subTest(overrides=overrides):
                    warm = self.configure(workspace, **overrides)
                    self.assertNotEqual(cold["key"], warm["key"])
                    self.assertEqual(cold["prefix"], warm["prefix"])
                    self.assertTrue(cold["key"].startswith(warm["prefix"]))
            for overrides in (
                {"variant": "release"},
                {"toolchain": "sha256:two"},
                {"RUNNER_ARCH": "ARM64"},
                {"image": "", "toolchain": "gcc 15; ccache 4.12"},
            ):
                with self.subTest(overrides=overrides):
                    incompatible = self.configure(workspace, **overrides)
                    self.assertFalse(cold["key"].startswith(incompatible["prefix"]))
            self.assertFalse('ccache | "old-job"'.startswith(cold["prefix"]))
            self.assertLess(len(cold["key"]), 512)
            self.assertEqual(RESTORE["with"]["restore-keys"].strip(), "${{ steps.configure.outputs.prefix }}")

    def test_missing_toolchain_fails_instead_of_sharing_fallback(self):
        with (
            patch("subprocess.check_output", side_effect=FileNotFoundError("docker")),
            patch.dict(os.environ, {"DOCKER_IMAGE": "missing-image"}),
            self.assertRaises(FileNotFoundError),
        ):
            exec(compile(CONFIGURE["run"], "configure-ccache", "exec"), {})

    def test_experiment_namespace_is_isolated_and_stable_across_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            main = self.configure(workspace)
            ref = "refs/heads/eserscor-2026/11/13-branch-ccache-experiment"
            cold = self.configure(workspace, GITHUB_REF=ref)
            warm = self.configure(workspace, GITHUB_REF=ref, GITHUB_RUN_ID="101")
            self.assertNotEqual(main["prefix"], cold["prefix"])
            self.assertEqual(cold["prefix"], warm["prefix"])
            self.assertNotEqual(cold["key"], warm["key"])

    def test_workflows_publish_only_successful_main_or_experiment(self):
        cache_jobs = 0
        for filename in WORKFLOWS:
            workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
            for job in workflow["jobs"].values():
                steps = job["steps"]
                setup = next((s for s in steps if s.get("uses") == "./.github/actions/setup-linux-ccache"), None)
                if not setup:
                    continue
                cache_jobs += 1
                self.assertNotIn("if", setup)
                self.assertNotIn("if", RESTORE)
                save = next(s for s in steps if s.get("name") == "Save CCache")
                self.assertEqual(save["with"]["path"], RESTORE["with"]["path"])
                self.assertFalse(Path(save["with"]["path"]).is_absolute())
                self.assertNotIn("..", Path(save["with"]["path"]).parts)
                self.assertIn("steps.ccache.outputs.cache-primary-key", save["with"]["key"])
                for ref, success, expected in (
                    ("refs/heads/main", True, True),
                    ("refs/heads/main", False, False),
                    ("refs/heads/eserscor-2026/11/13-branch-ccache-experiment", True, True),
                    ("refs/heads/eserscor-2026/11/13-branch-ccache-experiment", False, False),
                    ("refs/heads/unrelated", True, False),
                    ("refs/pull/42/merge", True, False),
                    ("refs/heads/rel-1.24", True, False),
                ):
                    condition = save["if"][3:-2].strip().replace("&&", "and").replace("||", "or")
                    condition = condition.replace("success()", str(success)).replace("github.ref", repr(ref))
                    self.assertEqual(eval(condition, {"__builtins__": {}}), expected)
                setup_index = steps.index(setup)
                if "docker-image" in setup["with"]:
                    image_index = next(i for i, s in enumerate(steps) if s.get("id") == "build_docker_image_step")
                    self.assertLess(image_index, setup_index)
                else:
                    tools_index = next(
                        i for i, s in enumerate(steps) if s.get("uses") == "./.github/actions/setup-build-tools"
                    )
                    self.assertLess(tools_index, setup_index)
                for step in steps:
                    if "run-build-script-in-docker@" in step.get("uses", ""):
                        self.assertIn("steps.ccache.outputs.container-prefix", step["with"]["python_path_prefix"])
                        self.assertLess(setup_index, steps.index(step))
                    if step.get("name") == "Run Build 7 (Using docker run)":
                        self.assertIn("-e CCACHE_DIR=/onnxruntime_src/.cache/ccache", step["run"])
                        self.assertIn("-e CCACHE_UMASK", step["run"])
        self.assertEqual(cache_jobs, 10)

    @unittest.skipUnless(sys.platform == "linux", "Requires bash")
    def test_container_prefix_preserves_python_setup_and_stops_on_failure(self):
        prefix = ACTION["outputs"]["container-prefix"]["value"]
        for python_prefix in ("true", "PATH=/custom/python:$PATH", "export ORT_TEST=1 PATH=/custom/python:$PATH"):
            command = (
                f"set -e; {prefix} && {python_prefix} && "
                'test "$CCACHE_DIR" = /onnxruntime_src/.cache/ccache && test "$CCACHE_UMASK" = 000'
            )
            if "PATH=" in python_prefix:
                command += ' && test "${PATH%%:*}" = /custom/python'
            if "ORT_TEST" in python_prefix:
                command += ' && test "$ORT_TEST" = 1'
            subprocess.run(["bash", "-c", command], check=True)
        result = subprocess.run(
            ["bash", "-c", f"set -e; {prefix} && false && echo unexpected"], check=False, capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(b"unexpected", result.stdout)

    @unittest.skipUnless(
        sys.platform == "linux" and shutil.which("ccache") and shutil.which("gcc"), "Linux ccache smoke"
    )
    def test_cold_population_and_warm_restore_at_different_workspace_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.c"
            source.write_text("int answer(void) { return 42; }\n")
            cold = root / "old/workspace"
            warm = root / "different/runner/work/checkout"
            for workspace in (cold, warm):
                workspace.mkdir(parents=True)
                self.configure(workspace, image="", toolchain=None)
                env_file = workspace / "env"
                env = {**os.environ, **dict(line.split("=", 1) for line in env_file.read_text().splitlines())}
                subprocess.run(["bash", "-c", ACCESS["run"]], env=env, check=True)
                if workspace == warm:
                    subprocess.run(["tar", "-xf", str(root / "cache.tar")], cwd=warm, check=True)
                    subprocess.run(["bash", "-c", ACCESS["run"]], env=env, check=True)
                subprocess.run(["ccache", "--zero-stats"], env=env, check=True, capture_output=True)
                subprocess.run(["ccache", "gcc", "-c", str(source), "-o", str(root / "source.o")], env=env, check=True)
                stats = dict(
                    line.split("\t")
                    for line in subprocess.check_output(["ccache", "--print-stats"], env=env, text=True).splitlines()
                )
                self.assertEqual(int(stats["cache_miss"]), int(workspace == cold))
                self.assertEqual(int(stats["direct_cache_hit"]), int(workspace == warm))
                if workspace == cold:
                    subprocess.run(
                        [
                            "tar",
                            "--posix",
                            "-cf",
                            str(root / "cache.tar"),
                            "-P",
                            "-C",
                            str(cold),
                            RESTORE["with"]["path"],
                        ],
                        check=True,
                    )
            members = subprocess.check_output(["tar", "-tf", str(root / "cache.tar")], text=True).splitlines()
            self.assertTrue(members)
            self.assertTrue(all(name.startswith(".cache/ccache/") and ".." not in name.split("/") for name in members))

    @unittest.skipUnless(
        sys.platform == "linux" and os.environ.get("ORT_CCACHE_DOCKER_IMAGE"),
        "Set ORT_CCACHE_DOCKER_IMAGE to an image containing bash, gcc and ccache",
    )
    def test_container_uids_populate_and_restore_portable_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cold = root / "old/workspace"
            warm = root / "new/deeper/runner/workspace"
            home_cache = root / "home-cache"
            home_cache.mkdir()
            for workspace, uid in ((cold, "1001:1001"), (warm, "1002:1002")):
                workspace.mkdir(parents=True)
                (workspace / "fixture.c").write_text(
                    "#ifndef ANSWER\n#define ANSWER 42\n#endif\nint answer(void) { return ANSWER; }\n"
                )
                self.configure(workspace, image=os.environ["ORT_CCACHE_DOCKER_IMAGE"], toolchain=None)
                env = {**os.environ, "CCACHE_DIR": str(workspace / ".cache/ccache")}
                if workspace == warm:
                    subprocess.run(["tar", "-xf", str(root / "cache.tar")], cwd=workspace, check=True)
                subprocess.run(["bash", "-c", ACCESS["run"]], env=env, check=True)
                prefix = ACTION["outputs"]["container-prefix"]["value"]
                result = subprocess.check_output(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--user",
                        uid,
                        "--volume",
                        f"{workspace}:/onnxruntime_src",
                        "--volume",
                        f"{home_cache}:/home/onnxruntimedev/.cache",
                        "-e",
                        "CCACHE_DIR=/home/onnxruntimedev/.cache/ccache",
                        os.environ["ORT_CCACHE_DOCKER_IMAGE"],
                        "bash",
                        "-c",
                        f"set -e; {prefix} && "
                        'test "$(ccache --get-config cache_dir)" = /onnxruntime_src/.cache/ccache && '
                        "ccache --zero-stats >/dev/null && "
                        "ccache gcc -c /onnxruntime_src/fixture.c -o /tmp/fixture.o && "
                        "ccache --print-stats && "
                        f"ccache gcc -DANSWER={uid.split(':')[0]} -c /onnxruntime_src/fixture.c -o /tmp/uid.o",
                    ],
                    text=True,
                )
                stats = dict(line.split("\t") for line in result.splitlines())
                self.assertEqual(int(stats["cache_miss"]), int(workspace == cold))
                self.assertEqual(int(stats["direct_cache_hit"]), int(workspace == warm))
                entries = [
                    path
                    for path in (workspace / ".cache/ccache").rglob("*R")
                    if path.stat().st_uid == int(uid.split(":")[0])
                ]
                self.assertTrue(entries)
                for entry in entries:
                    with entry.open("r+b") as output:
                        self.assertTrue(output.writable())
                self.assertEqual(list(home_cache.iterdir()), [])
                if workspace == cold:
                    subprocess.run(
                        [
                            "tar",
                            "--posix",
                            "-cf",
                            str(root / "cache.tar"),
                            "-P",
                            "-C",
                            str(cold),
                            RESTORE["with"]["path"],
                        ],
                        check=True,
                    )
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--user",
                    "1001:1001",
                    "--volume",
                    f"{warm}:/onnxruntime_src",
                    "-e",
                    "CCACHE_DIR=/onnxruntime_src/.cache/ccache",
                    "-e",
                    "CCACHE_UMASK=000",
                    os.environ["ORT_CCACHE_DOCKER_IMAGE"],
                    "bash",
                    "-c",
                    'test "$(ccache --get-config cache_dir)" = /onnxruntime_src/.cache/ccache && '
                    "ccache gcc -DANSWER=1003 -c /onnxruntime_src/fixture.c -o /tmp/direct.o",
                ],
                check=True,
            )
            for entry in (warm / ".cache/ccache").rglob("*R"):
                with entry.open("r+b") as output:
                    self.assertTrue(output.writable())


if __name__ == "__main__":
    unittest.main()
