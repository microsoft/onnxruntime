# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import update_webgpu_cgmanifest as updater
import validate_webgpu_cgmanifest as validator

OLD = "1" * 40
DAWN = "2" * 40
CHILD = "3" * 40
TAG_OBJECT = "4" * 40
REPO = "https://github.com/microsoft/DirectXShaderCompiler.git"
PATH = "third_party/directx-shader-compiler/src"
SOURCE = f"""
vars = {{'chromium_git': 'https://chromium.googlesource.com', 'revision': '{CHILD}', 'version': Str('3.0')}}
deps = {{
    '{PATH}': {{
        'url': '{{chromium_git}}/external/github.com/microsoft/DirectXShaderCompiler.git@{{revision}}',
        'condition': 'host_os == "win"',
    }},
    'tools/win': Var('chromium_git') + '/chromium/src/tools/win@{CHILD}',
    'tools/cipd': {{'dep_type': 'cipd', 'packages': [{{'version': Var('version')}}]}},
    'tools/gcs': {{'dep_type': 'gcs', 'objects': []}},
    'removed': None,
}}
"""


def make_manifest():
    return {
        "version": 1,
        "registrations": [
            {
                "component": {
                    "type": "git",
                    "git": {"repositoryUrl": validator.DAWN_REPOSITORY_URL, "commitHash": OLD},
                },
                "comments": "runtime; Dawn root",
            },
            {
                "component": {"type": "git", "git": {"repositoryUrl": REPO, "commitHash": OLD, "tag": "v1.8.2502"}},
                "comments": "runtime; separately packaged DXC release; SHA256: keep-me",
            },
            {
                "component": {"type": "git", "git": {"repositoryUrl": REPO, "commitHash": OLD}},
                "comments": "build-tool; Dawn DEPS third_party/dxc. Reviewed classification.",
                "developmentDependency": True,
                "detectedComponentLocations": ["{SourceFileRoot}/cmake/deps.txt"],
                "dependencyRoots": [
                    {"type": "git", "git": {"repositoryUrl": validator.DAWN_REPOSITORY_URL, "commitHash": OLD}}
                ],
            },
        ],
    }


class DawnDepsTests(unittest.TestCase):
    def test_parse_all_git_pins_including_conditional_entries(self):
        deps = updater.parse_dawn_deps(SOURCE)
        self.assertEqual(set(deps), {PATH, "tools/win"})
        self.assertEqual(deps[PATH]["commitHash"], CHILD)
        self.assertEqual(deps[PATH]["condition"], 'host_os == "win"')
        self.assertEqual(deps["tools/win"]["repositoryUrl"], "https://chromium.googlesource.com/chromium/src/tools/win")

    def test_does_not_execute_deps_or_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            code = f"__import__('pathlib').Path({str(marker)!r}).touch()"
            self.assertEqual(updater.parse_dawn_deps(SOURCE + "\n" + code), updater.parse_dawn_deps(SOURCE))
            with self.assertRaises(ValueError):
                updater.parse_dawn_deps(f"vars = {{}}\ndeps = {{'malicious': {code}}}")
            self.assertFalse(marker.exists())

    def test_rejects_unsupported_or_ambiguous_metadata(self):
        sources = [
            SOURCE.replace(CHILD, "main"),
            SOURCE.replace("{revision}", "{missing}"),
            SOURCE + "\ndeps = {}",
            "vars = {}\ndeps = {}",
            "vars = {}\ndeps = dict()",
            "vars = {}\ndeps = {'a': None, 'a': None}",
            "vars = {}\ndeps = {",
        ]
        for source in sources:
            with self.subTest(source=source), self.assertRaises((ValueError, KeyError, SyntaxError)):
                updater.parse_dawn_deps(source)

    def test_upstream_mirrors_and_split_llvm_repository_identities(self):
        aliases = [
            (
                "https://chromium.googlesource.com/external/github.com/Google/googletest",
                "https://github.com/google/googletest.git",
            ),
            (
                "https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/",
                "https://gitlab.freedesktop.org/mesa/mesa.git",
            ),
        ]
        for mirror, upstream in aliases:
            with self.subTest(mirror=mirror):
                self.assertEqual(updater._repository_identity(mirror), updater._repository_identity(upstream))
        llvm = "https://chromium.googlesource.com/external/github.com/llvm/llvm-project/libcxx"
        self.assertEqual(updater._repository_identity(llvm + ".git"), llvm)

    def test_resolves_lightweight_and_annotated_tags(self):
        for refs in [f"{DAWN}\trefs/tags/v1\n", f"{TAG_OBJECT}\trefs/tags/v1\n{DAWN}\trefs/tags/v1^{{}}\n"]:
            with self.subTest(refs=refs), patch.object(updater.subprocess, "run") as run:
                run.return_value.stdout = refs
                self.assertEqual(updater._resolve_dawn_commit("tag", "v1"), DAWN)
                self.assertEqual(run.call_args.args[0][-2:], ["refs/tags/v1", "refs/tags/v1^{}"])
        with patch.object(updater.subprocess, "run") as run:
            self.assertEqual(updater._resolve_dawn_commit("commitHash", DAWN), DAWN)
            run.assert_not_called()
        with patch.object(updater.subprocess, "run") as run:
            run.return_value.stdout = ""
            with self.assertRaisesRegex(ValueError, "could not resolve"):
                updater._resolve_dawn_commit("tag", "missing")

    def test_download_uses_resolved_commit(self):
        with patch.object(updater.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = SOURCE.encode()
            self.assertEqual(updater._download_deps(DAWN), SOURCE)
            urlopen.assert_called_once_with(f"https://raw.githubusercontent.com/google/dawn/{DAWN}/DEPS", timeout=30)


class ManifestTests(unittest.TestCase):
    def test_updates_hashes_and_renamed_path_preserving_reviewed_metadata(self):
        original = make_manifest()
        snapshot = copy.deepcopy(original)
        updated = updater.update_manifest(original, DAWN, "v1", updater.parse_dawn_deps(SOURCE))
        expected = copy.deepcopy(original)
        root, _, child = expected["registrations"]
        root["component"]["git"].update(commitHash=DAWN, tag="v1")
        child["component"]["git"]["commitHash"] = CHILD
        child["comments"] = child["comments"].replace("third_party/dxc", PATH)
        child["dependencyRoots"][0]["git"]["commitHash"] = DAWN
        self.assertEqual(updated, expected)
        self.assertEqual(original, snapshot)
        self.assertEqual(updater.update_manifest(updated, DAWN, "v1", updater.parse_dawn_deps(SOURCE)), updated)

    def test_commit_pin_removes_old_tag(self):
        manifest = make_manifest()
        manifest["registrations"][0]["component"]["git"]["tag"] = "old-tag"
        updated = updater.update_manifest(manifest, DAWN, None, updater.parse_dawn_deps(SOURCE))
        self.assertNotIn("tag", updated["registrations"][0]["component"]["git"])

    def test_rejects_removed_changed_or_ambiguous_repositories(self):
        deps = updater.parse_dawn_deps(SOURCE)
        variants = [
            {"tools/win": deps["tools/win"]},
            {"third_party/dxc": deps["tools/win"], PATH: deps[PATH]},
            {PATH: deps[PATH], "another/path": deps[PATH]},
        ]
        for dependencies in variants:
            with self.subTest(dependencies=dependencies), self.assertRaisesRegex(ValueError, "No manifest written"):
                updater.update_manifest(make_manifest(), DAWN, None, dependencies)

    def test_duplicate_repository_is_disambiguated_by_path(self):
        deps = updater.parse_dawn_deps(SOURCE)
        deps["third_party/dxc"] = dict(deps[PATH], commitHash=TAG_OBJECT)
        updated = updater.update_manifest(make_manifest(), DAWN, None, deps)
        self.assertEqual(updated["registrations"][2]["component"]["git"]["commitHash"], TAG_OBJECT)


class CommandTests(unittest.TestCase):
    def setUp(self):
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        directory = contexts.enter_context(tempfile.TemporaryDirectory())
        self.manifest = Path(directory) / "manifest.json"
        self.manifest.write_text(json.dumps(make_manifest(), indent=2) + "\n", encoding="utf-8")
        self.deps_file = Path(directory) / "deps.txt"
        self.deps_file.write_text(
            f"dawn;https://github.com/google/dawn/archive/{DAWN}.zip;archive-hash\n", encoding="utf-8"
        )
        self.download = contexts.enter_context(patch.object(updater, "_download_deps", return_value=SOURCE))
        self.stdout = contexts.enter_context(redirect_stdout(io.StringIO()))
        self.stderr = contexts.enter_context(redirect_stderr(io.StringIO()))
        self.args = ["--manifest", str(self.manifest), "--deps-file", str(self.deps_file)]

    def test_check_update_check_and_idempotence(self):
        before = self.manifest.read_bytes()
        self.assertEqual(updater.main([*self.args, "--check"]), 1)
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertEqual(updater.main(self.args), 0)
        after = self.manifest.read_bytes()
        self.assertNotEqual(after, before)
        self.assertEqual(updater.main([*self.args, "--check"]), 0)
        self.assertEqual(updater.main(self.args), 0)
        self.assertEqual(self.manifest.read_bytes(), after)

    def test_failures_do_not_write_partial_manifest(self):
        before = self.manifest.read_bytes()
        for error in [OSError("network failed"), subprocess.TimeoutExpired("git", 30)]:
            with self.subTest(error=error):
                self.download.side_effect = error
                self.assertEqual(updater.main(self.args), 1)
                self.assertEqual(self.manifest.read_bytes(), before)
        self.download.side_effect = None
        for source in ["invalid DEPS!", SOURCE.replace("DirectXShaderCompiler", "different-repo")]:
            with self.subTest(source=source):
                self.download.return_value = source
                self.assertEqual(updater.main(self.args), 1)
                self.assertEqual(self.manifest.read_bytes(), before)

    def test_list_all_deps_without_a_manifest(self):
        self.manifest.unlink()
        self.assertEqual(updater.main([*self.args, "--list-deps"]), 0)
        self.assertEqual(json.loads(self.stdout.getvalue()), updater.parse_dawn_deps(SOURCE))
        self.assertFalse(self.manifest.exists())

    def test_tag_pin_in_deps_txt(self):
        self.deps_file.write_text(
            "dawn;https://github.com/google/dawn/archive/refs/tags/v1.zip;hash\n", encoding="utf-8"
        )
        with patch.object(updater.subprocess, "run") as run:
            run.return_value.stdout = f"{DAWN}\trefs/tags/v1\n"
            self.assertEqual(updater.main(self.args), 0)
        self.download.assert_called_once_with(DAWN)
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["registrations"][0]["component"]["git"]["tag"], "v1")
        with patch.object(validator, "_dawn_pin_from_deps_txt", return_value=("tag", "v1")):
            validator._validate_dawn_root(manifest)
            validator._validate_dawn_dependency_roots(manifest)

    def test_rejects_missing_or_duplicate_dawn_pins(self):
        pin = self.deps_file.read_text(encoding="utf-8")
        for text in ["", pin + pin, pin.replace(DAWN, "main")]:
            with self.subTest(text=text):
                self.deps_file.write_text(text, encoding="utf-8")
                self.assertEqual(updater.main(self.args), 1)
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
