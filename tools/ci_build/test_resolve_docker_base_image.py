# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import io
import json
import os
import shlex
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import resolve_docker_base_image as resolver


class ResolveDockerBaseImageTest(unittest.TestCase):
    def setUp(self):
        self.manifest = resolver.load_manifest()
        self.family = "cuda13_x64_almalinux8_gcc14"
        image = self.manifest["images"][self.family]
        self.reference = f"{image['repository']}@{image['digest']}"
        self.dockerfile = str(resolver.DOCKER_DIR / "Dockerfile.manylinux2_28_cuda")

    def test_explicit_family_resolves_to_digest_not_tag(self):
        self.assertEqual(resolver.resolve_family(self.family), self.reference)
        self.assertNotIn(":20261104.1", self.reference)

    def test_all_dockerfile_defaults_resolve_and_have_no_embedded_pin(self):
        for dockerfile, family in self.manifest["dockerfile_defaults"].items():
            path = resolver.DOCKER_DIR / dockerfile
            with self.subTest(dockerfile=dockerfile):
                self.assertEqual(
                    resolver.resolve_build_args(str(path)),
                    ["--build-arg", f"BASEIMAGE={resolver.resolve_family(family)}"],
                )
                text = path.read_text(encoding="utf-8")
                self.assertIn("ARG BASEIMAGE\n", text)
                self.assertNotIn("ARG BASEIMAGE=", text)

    def test_relative_and_absolute_dockerfile_paths_use_same_default(self):
        relative = os.path.relpath(self.dockerfile)
        self.assertEqual(resolver.resolve_build_args(str(relative)), resolver.resolve_build_args(self.dockerfile))

    def test_family_override_preserves_other_build_arguments(self):
        result = resolver.resolve_build_args(
            self.dockerfile, '--network=host --build-arg TRT_VERSION= --build-arg "LABEL=two words"', self.family
        )
        self.assertEqual(
            result,
            [
                "--network=host",
                "--build-arg",
                "TRT_VERSION=",
                "--build-arg",
                "LABEL=two words",
                "--build-arg",
                f"BASEIMAGE={self.reference}",
            ],
        )
        self.assertEqual(shlex.split(shlex.join(result)), result)

    def test_explicit_base_image_bypasses_default(self):
        for arguments in (
            "--build-arg BASEIMAGE=custom/image:tag",
            "--build-arg=BASEIMAGE=custom/image@sha256:" + "a" * 64,
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(resolver.resolve_build_args(self.dockerfile, arguments), shlex.split(arguments))

    def test_ambiguous_or_empty_base_arguments_fail(self):
        for arguments, family in (
            ("--build-arg BASEIMAGE=custom/image:tag", self.family),
            ("--build-arg BASEIMAGE=a --build-arg=BASEIMAGE=b", ""),
            ("--build-arg BASEIMAGE=", ""),
            ("--build-arg BASEIMAGE", ""),
        ):
            with self.subTest(arguments=arguments, family=family), self.assertRaises(ValueError):
                resolver.resolve_build_args(self.dockerfile, arguments, family)

    def test_unknown_family_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown Docker base-image family"):
            resolver.resolve_build_args(self.dockerfile, family="missing-family")

    def test_unmanaged_dockerfile_does_not_get_an_unrelated_default(self):
        arguments = "--build-arg EXISTING=value"
        self.assertEqual(resolver.resolve_build_args("Dockerfile.unmanaged", arguments), shlex.split(arguments))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Dockerfile.manylinux2_28_cuda"
            self.assertEqual(resolver.resolve_build_args(str(path)), [])

    def test_manifest_digest_change_changes_resolved_cache_input(self):
        before = resolver.resolve_build_args(self.dockerfile, family=self.family)
        changed = copy.deepcopy(self.manifest)
        changed["images"][self.family]["digest"] = "sha256:" + "a" * 64
        with patch.object(resolver, "load_manifest", return_value=changed):
            after = resolver.resolve_build_args(self.dockerfile, family=self.family)
        self.assertNotEqual(before, after)
        self.assertIn("BASEIMAGE=" + changed["images"][self.family]["repository"] + "@sha256:" + "a" * 64, after)

    def test_invalid_manifest_entries_fail(self):
        for field, value in (
            ("digest", ""),
            ("digest", "sha256:not-a-digest"),
            ("repository", "registry/image:mutable"),
            ("tag", "tag\ninjection"),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                manifest = copy.deepcopy(self.manifest)
                manifest["images"][self.family][field] = value
                path = Path(directory) / "base-images.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with patch.object(resolver, "MANIFEST_PATH", path), self.assertRaises(ValueError):
                    resolver.load_manifest()

    def test_missing_digest_or_unknown_default_fails(self):
        for missing_digest in (True, False):
            with self.subTest(missing_digest=missing_digest), tempfile.TemporaryDirectory() as directory:
                manifest = copy.deepcopy(self.manifest)
                if missing_digest:
                    del manifest["images"][self.family]["digest"]
                else:
                    manifest["dockerfile_defaults"]["Dockerfile.test"] = "missing-family"
                path = Path(directory) / "base-images.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with patch.object(resolver, "MANIFEST_PATH", path), self.assertRaises((KeyError, ValueError)):
                    resolver.load_manifest()

    def test_legacy_arm64_family_stays_on_almalinux9_and_is_digest_pinned(self):
        family = "cuda13_aarch64_almalinux9_gcc14"
        image = self.manifest["images"][family]
        self.assertEqual(image["tag"], "20260323.1")
        self.assertIn("/public/azureml/onnxruntime_build_cuda13_aarch64_almalinux9_", image["repository"])
        self.assertEqual(resolver.resolve_family(family), f"{image['repository']}@{image['digest']}")

    def test_cli_prints_reference_for_packaging_metadata(self):
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["resolve", "--family", self.family]), redirect_stdout(stdout):
            resolver.main()
        self.assertEqual(stdout.getvalue(), self.reference + "\n")

    def test_cli_writes_resolved_github_build_args(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            output.write_text("existing=value\n", encoding="utf-8")
            arguments = [
                "resolve",
                "--dockerfile",
                self.dockerfile,
                "--family",
                self.family,
                "--build-args=--build-arg TRT_VERSION=",
                "--github-output",
                str(output),
            ]
            with patch.object(sys, "argv", arguments):
                resolver.main()
            lines = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "existing=value")
        self.assertEqual(
            shlex.split(lines[1].removeprefix("build-args=")),
            ["--build-arg", "TRT_VERSION=", "--build-arg", f"BASEIMAGE={self.reference}"],
        )

    def test_cli_errors_do_not_write_success_output(self):
        for extra in (
            ["--family", "missing-family"],
            ["--build-args=--build-arg 'LABEL=line\ninjection'"],
        ):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "github-output"
                arguments = ["resolve", "--dockerfile", self.dockerfile, "--github-output", str(output), *extra]
                with (
                    patch.object(sys, "argv", arguments),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    resolver.main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(output.exists())

    def test_pins_are_not_duplicated_in_ci_or_dockerfiles(self):
        root = Path(__file__).resolve().parents[2]
        for folder in (root / ".github", root / "tools/ci_build/github"):
            for path in folder.rglob("*"):
                if not path.is_file() or (
                    path.suffix not in (".yml", ".yaml") and not path.name.startswith("Dockerfile")
                ):
                    continue
                with self.subTest(path=path):
                    text = path.read_text(encoding="utf-8")
                    for image in self.manifest["images"].values():
                        self.assertNotIn(image["repository"], text)

    def test_workflows_do_not_bypass_the_shared_docker_action(self):
        root = Path(__file__).resolve().parents[2]
        for path in (root / ".github/workflows").glob("*.yml"):
            active_lines = [
                line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")
            ]
            with self.subTest(path=path):
                self.assertNotIn("/build-docker-image@", "\n".join(active_lines))
        wrapper = (root / ".github/actions/build-docker-image/action.yml").read_text(encoding="utf-8")
        self.assertIn("build-args: ${{ steps.resolve.outputs.build-args }}", wrapper)
        self.assertLess(wrapper.index("id: resolve"), wrapper.index("uses: microsoft/onnxruntime-github-actions/"))


if __name__ == "__main__":
    unittest.main()
