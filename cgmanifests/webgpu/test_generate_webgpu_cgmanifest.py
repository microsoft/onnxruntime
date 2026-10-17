# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import generate_webgpu_cgmanifest as generator


class GenerateManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "dawn"
        (self.source / "tools").mkdir(parents=True)
        self.seed = self.root / "seed"
        self.seed.mkdir()
        self.git(self.seed, "init")
        self.git(
            self.seed,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=",
            "commit",
            "--allow-empty",
            "-m",
            "fixture",
        )
        self.sha = self.git(self.seed, "rev-parse", "HEAD")
        self.pin = self.root / "deps.txt"
        self.pin.write_text("dawn;https://github.com/google/dawn/archive/refs/tags/v-test.zip;" + "a" * 40 + "\n")
        self.path = "third_party/new-dependency/src"
        self.checkout(self.path)
        self.write_dawn([self.path], {self.path: self.url("new-dependency")})

    @staticmethod
    def git(source, *args):
        return subprocess.run(
            ["git", "-C", str(source), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def checkout(self, path):
        destination = self.source / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", str(self.seed), str(destination)], check=True, capture_output=True)
        # Dawn's shallow fetch uses git init/fetch and need not have an origin.
        self.git(destination, "remote", "remove", "origin")

    def url(self, name):
        return "{chromium_git}/external/github.com/example/" + name + "@" + self.sha

    def write_dawn(self, paths, urls):
        fetcher = "def main(args):\n    required_submodules = " + repr(paths) + "\n"
        (self.source / "tools/fetch_dawn_dependencies.py").write_text(fetcher)
        deps = {path: {"url": url} for path, url in urls.items()}
        (self.source / "DEPS").write_text(
            "vars = {'chromium_git': 'https://chromium.googlesource.com'}\ndeps = " + repr(deps) + "\n"
        )

    def manifest(self):
        return generator.generate(self.source, self.pin)

    def test_new_upstream_dependency_is_discovered_without_ort_list(self):
        added = "third_party/future-library/src"
        self.checkout(added)
        self.write_dawn([self.path, added], {self.path: self.url("new-dependency"), added: self.url("future")})
        registrations = self.manifest()["registrations"]
        self.assertEqual(len(registrations), 3)
        git_entries = [r["component"]["git"] for r in registrations[1:]]
        self.assertIn({"repositoryUrl": "https://github.com/example/future", "commitHash": self.sha}, git_entries)

    def test_unfetched_deps_entries_are_not_included(self):
        self.write_dawn([self.path], {self.path: self.url("new-dependency"), "unused": self.url("unused")})
        self.assertEqual(len(self.manifest()["registrations"]), 2)

    def test_nested_fetched_dependency_without_vars_is_included(self):
        nested = self.path + "/nested"
        self.checkout(nested)
        self.write_dawn([self.path, nested], {self.path: self.url("parent")})
        (self.source / self.path / "DEPS").write_text(
            "deps = {'nested': {'url': 'https://github.com/example/nested@" + self.sha + "'}}\n"
        )
        registrations = self.manifest()["registrations"]
        self.assertEqual(len(registrations), 3)
        self.assertIn(
            {"repositoryUrl": "https://github.com/example/nested", "commitHash": self.sha},
            [r["component"]["git"] for r in registrations[1:]],
        )

    def test_missing_checkout_fails(self):
        self.write_dawn(["missing"], {"missing": self.url("missing")})
        with self.assertRaisesRegex(ValueError, "Missing fetched dependency"):
            self.manifest()

    def test_stale_checkout_fails(self):
        self.write_dawn([self.path], {self.path: self.url("new-dependency").replace(self.sha, "b" * 40)})
        with self.assertRaisesRegex(ValueError, "Stale dependency"):
            self.manifest()

    def test_metadata_is_not_executed(self):
        for path in [self.source / "DEPS", self.source / "tools/fetch_dawn_dependencies.py"]:
            with path.open("a") as output:
                output.write("\nraise RuntimeError('must not execute')\n")
        self.assertEqual(len(self.manifest()["registrations"]), 2)

    def test_changed_upstream_list_syntax_fails(self):
        (self.source / "tools/fetch_dawn_dependencies.py").write_text("required_submodules = compute_dependencies()")
        with self.assertRaises(ValueError):
            self.manifest()

    def test_duplicate_components_are_deduplicated(self):
        self.checkout("second-copy")
        self.write_dawn([self.path, "second-copy"], {self.path: self.url("same"), "second-copy": self.url("same")})
        self.assertEqual(len(self.manifest()["registrations"]), 2)

    def test_var_and_concatenated_url_are_supported(self):
        (self.source / "DEPS").write_text(
            "vars = {'repo': 'https://github.com/example/lib', 'revision': '" + self.sha + "'}\n"
            "deps = {'" + self.path + "': {'url': Var('repo') + '@' + Var('revision')}}\n"
        )
        self.assertEqual(self.manifest()["registrations"][1]["component"]["git"]["commitHash"], self.sha)

    def test_chromium_str_variables_are_supported(self):
        path = self.source / "DEPS"
        path.write_text(path.read_text().replace("vars = {", "vars = {'android_version': Str('2@30.0'), "))
        self.assertEqual(len(self.manifest()["registrations"]), 2)

    def test_dxc_release_is_separate_and_opt_in(self):
        pipeline = self.root / "windows.yml"
        pipeline.write_text(
            '$dxcZipUrl = "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v-next/dxc.zip"\n'
            '$expectedHash = "' + "C" * 64 + '"\n'
        )
        manifest = generator.generate(self.source, self.pin, pipeline)
        self.assertEqual(len(manifest["registrations"]), 3)
        release = manifest["registrations"][-1]["component"]["other"]
        self.assertEqual(release["version"], "v-next")
        self.assertEqual(release["hash"], "C" * 64)
        self.assertEqual(len(self.manifest()["registrations"]), 2)

    def test_archive_pin_and_cli_output_are_reproducible(self):
        outputs = [self.root / "staged/cgmanifest.json", self.root / "artifact/cgmanifest.json"]
        command = [sys.executable, generator.__file__, "--dawn-source", str(self.source), "--deps-file", str(self.pin)]
        for output in outputs:
            command.extend(["--output", str(output)])
        subprocess.run(command, check=True, capture_output=True)
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        first = outputs[0].read_bytes()
        subprocess.run(command, check=True, capture_output=True)
        self.assertEqual(first, outputs[0].read_bytes())
        archive = json.loads(first)["registrations"][0]["component"]["other"]
        self.assertEqual(archive["version"], "v-test")
        self.assertEqual(archive["hash"], "a" * 40)


if __name__ == "__main__":
    unittest.main()
