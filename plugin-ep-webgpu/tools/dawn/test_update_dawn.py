# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import difflib
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import patch_dawn
import update_dawn as updater

OLD = "1" * 40
NEW = "2" * 40
CHILD = "3" * 40
DEP = "https://github.com/example/dependency.git"
SWIFT = "https://swiftshader.googlesource.com/SwiftShader"


def deps_text(commit):
    return f"""vars = {{}}
deps = {{
    'third_party/kept': {{'url': '{DEP}@{commit}'}},
    'third_party/swiftshader': {{'url': '{SWIFT}@{commit}'}},
}}
"""


def registration(url, commit, comments, development=False):
    result = {"component": {"type": "git", "git": {"repositoryUrl": url, "commitHash": commit}}, "comments": comments}
    if development:
        result["developmentDependency"] = True
    return result


def manifest():
    return {
        "version": 1,
        "registrations": [
            registration(updater.validator.DAWN_REPOSITORY_URL, OLD, "Dawn root"),
            registration(DEP, OLD, "runtime; Dawn DEPS third_party/kept."),
        ],
    }


class FakeSource:
    def __init__(self, commit=NEW, cache=Path("."), files=None):
        self.commit = commit
        self.cache = cache
        self.files = {
            "DEPS": deps_text(CHILD),
            updater.FETCHER: "def main():\n    required_submodules = ['third_party/kept']\n",
        }
        if files:
            self.files.update(files)

    def read(self, path):
        return self.files[path]


class TempTest(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.root = Path(self.contexts.enter_context(tempfile.TemporaryDirectory()))

    def write(self, path, text):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(text.encode())


class PlanTests(TempTest):
    def test_dry_run_apply_preserves_unrelated_content_and_line_endings(self):
        self.write("file.txt", "old\r\nunrelated\r\n")
        plan = updater.Plan(self.root)
        plan.edit("file.txt", "new\nunrelated\n")
        self.assertEqual((self.root / "file.txt").read_bytes(), b"old\r\nunrelated\r\n")
        self.assertIn("+new\n", plan.diff())
        plan.apply()
        self.assertEqual((self.root / "file.txt").read_bytes(), b"new\r\nunrelated\r\n")

    def test_rejects_concurrent_edit_and_blockers_before_writing(self):
        self.write("file.txt", "old\n")
        plan = updater.Plan(self.root)
        plan.edit("file.txt", "new\n")
        self.write("file.txt", "user edit\n")
        with self.assertRaisesRegex(ValueError, "changed while preparing"):
            plan.apply()
        self.assertEqual((self.root / "file.txt").read_text(), "user edit\n")
        plan.report["blockers"].append("unreviewed component")
        with self.assertRaisesRegex(ValueError, "unresolved blockers"):
            plan.apply()

    def test_restores_written_files_on_write_failure(self):
        self.write("first", "old1")
        self.write("second", "old2")
        plan = updater.Plan(self.root)
        plan.edit("first", "new1")
        plan.edit("second", "new2")
        write_bytes = Path.write_bytes

        def fail_second(path, data):
            if path.name == "second" and data == b"new2":
                raise OSError("disk error")
            return write_bytes(path, data)

        with patch.object(Path, "write_bytes", fail_second), self.assertRaises(OSError):
            plan.apply()
        self.assertEqual((self.root / "first").read_text(), "old1")
        self.assertEqual((self.root / "second").read_text(), "old2")


class MetadataTests(TempTest):
    def test_fetcher_is_not_executed(self):
        source = "raise RuntimeError('do not execute')\ndef main():\n    required_submodules = ['a']\n"
        self.assertEqual(updater.required_paths(source), ["a"])
        for invalid in ["required_submodules = []", "required_submodules = get_paths()", "x = []"]:
            with self.subTest(source=invalid), self.assertRaises(ValueError):
                updater.required_paths(invalid)

    def test_inventory_reports_additions_removals_and_conditions(self):
        old = {"removed": {}, "same": {"condition": "old"}}
        new = {"added": {}, "same": {"condition": "new"}}
        self.assertEqual(
            updater.dependency_diff(old, new),
            {
                "added": {"added": {}},
                "removed": {"removed": {}},
                "changed": {"same": {"old": {"condition": "old"}, "new": {"condition": "new"}}},
            },
        )

    def test_removes_only_verified_obsolete_development_registrations(self):
        obsolete = registration("https://github.com/old/dep", OLD, "build-tool; Dawn DEPS old/path.", True)
        original = manifest()
        original["registrations"].append(obsolete)
        source = FakeSource(cache=self.root)
        prior = FakeSource(
            files={
                "DEPS": deps_text(OLD).replace(
                    "deps = {", "deps = {'old/path': 'https://github.com/old/dep@" + OLD + "',"
                )
            }
        )
        report = {"blockers": []}
        with patch.object(updater, "Source", return_value=prior):
            updated = updater.refresh_manifest(
                original, source, "v1", updater.cg.parse_dawn_deps(source.read("DEPS")), report
            )
            self.assertEqual(len(updated["registrations"]), 2)
            self.assertEqual(report["removedDevelopmentRegistrations"], [obsolete])
            self.assertEqual(len(original["registrations"]), 3)
            original["registrations"][-1].pop("developmentDependency")
            with self.assertRaisesRegex(ValueError, "cannot uniquely match"):
                updater.refresh_manifest(
                    original, source, "v1", updater.cg.parse_dawn_deps(source.read("DEPS")), {"blockers": []}
                )

    def test_new_required_component_blocks_instead_of_guessing_classification(self):
        source = FakeSource(files={updater.FETCHER: "required_submodules = ['third_party/new']"})
        report = {"blockers": []}
        with patch.object(updater, "Source", return_value=source):
            updater.refresh_manifest(manifest(), source, None, updater.cg.parse_dawn_deps(source.read("DEPS")), report)
        self.assertIn("third_party/new", report["blockers"][0])

    def test_same_repository_at_a_different_revision_requires_registration(self):
        source = FakeSource(
            files={
                "DEPS": deps_text(CHILD).replace(
                    "deps = {", f"deps = {{'third_party/other': {{'url': '{DEP}@{NEW}'}},"
                ),
                updater.FETCHER: "required_submodules = ['third_party/kept', 'third_party/other']",
            }
        )
        report = {"blockers": []}
        with patch.object(updater, "Source", return_value=source):
            updater.refresh_manifest(manifest(), source, None, updater.cg.parse_dawn_deps(source.read("DEPS")), report)
        self.assertIn("third_party/other", report["blockers"][0])
        self.assertIn("third_party/other", report["unregisteredGitDependencies"])

    def test_archive_reads_only_requested_file_and_rejects_ambiguous_paths(self):
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("dawn/DEPS", "metadata")
            output.writestr("dawn/../../escape", "bad")
        source = updater.Source(NEW, self.root, archive)
        self.assertEqual(source.read("DEPS"), "metadata")
        with self.assertRaises(ValueError):
            source.read("../../escape")
        with self.assertRaises(ValueError):
            source.read("missing")
        self.assertFalse((self.root / "escape").exists())
        with zipfile.ZipFile(archive, "a") as output:
            output.writestr("other/DEPS", "duplicate")
        with self.assertRaises(ValueError):
            source.read("DEPS")
        with self.assertRaises(ValueError):
            updater.Source("../../outside", self.root)

    def test_download_failure_leaves_no_partial_file(self):
        with patch.object(updater.urllib.request, "urlopen") as request:
            request.return_value.__enter__.return_value.read.side_effect = [b"some bytes", OSError("network")]
            with self.assertRaises(OSError):
                updater.download("https://example.com/archive.zip", self.root / "file.zip")
        self.assertEqual(list(self.root.iterdir()), [])

    def archive_download(self, url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("/DEPS"):
            destination.write_text(deps_text(CHILD), encoding="utf-8")
        else:
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("dawn/DEPS", deps_text(CHILD))

    def test_archive_hash_and_cache_integrity(self):
        with patch.object(updater, "download", side_effect=self.archive_download) as download:
            source, url, checksum, tag = updater.archive_for_revision(NEW, self.root)
            self.assertEqual(checksum, updater.digest(source.archive))
            self.assertEqual(url, f"https://github.com/google/dawn/archive/{NEW}.zip")
            self.assertIsNone(tag)
            updater.archive_for_revision(NEW, self.root)
            self.assertEqual(download.call_count, 2)  # Archive and pinned metadata, once each.
            source.archive.write_bytes(b"corrupt cache")
            with self.assertRaisesRegex(ValueError, "archive is corrupt"):
                updater.archive_for_revision(NEW, self.root)

    def test_tag_move_during_download_is_rejected(self):
        with (
            patch.object(updater, "download", side_effect=self.archive_download),
            patch.object(updater.cg, "_resolve_dawn_commit", side_effect=[NEW, OLD]),
            self.assertRaisesRegex(ValueError, "moved during"),
        ):
            updater.archive_for_revision("v1", self.root)

    def test_archive_metadata_must_match_resolved_commit(self):
        def wrong_metadata(url, destination):
            self.archive_download(url, destination)
            if url.endswith("/DEPS"):
                destination.write_text(deps_text(OLD), encoding="utf-8")

        with (
            patch.object(updater, "download", side_effect=wrong_metadata),
            self.assertRaisesRegex(ValueError, "does not match the resolved commit"),
        ):
            updater.archive_for_revision(NEW, self.root)


@unittest.skipUnless(shutil.which("patch"), "GNU patch is required")
class PatchTests(TempTest):
    def test_refresh_and_ordered_application(self):
        # The second patch depends on the first patch, and both have stale line offsets.
        before = "a\nb\nc\nd\ne\nf\ng\n"
        middle = before.replace("d\n", "changed\n")
        after = middle.replace("changed\n", "final\n")
        names = []
        for name, old, new in [("one", before, middle), ("two", middle, after)]:
            text = "".join(
                difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile="a/file", tofile="b/file")
            )
            self.write(f"cmake/patches/dawn/{name}.patch", text)
            names.append("${PROJECT_SOURCE_DIR}/patches/dawn/" + name + ".patch")
        self.write(
            "external.cmake",
            "set(ONNXRUNTIME_Dawn_PATCH_COMMAND\n" + "\n".join(names) + ")\nonnxruntime_fetchcontent_declare(",
        )
        plan = updater.Plan(self.root)
        source = FakeSource(files={"file": "prefix\n" + before})
        patch_dawn.refresh_patches(plan, source, "external.cmake", "patch")
        self.assertFalse(plan.report["blockers"])
        self.assertEqual([p["status"] for p in plan.report["patches"]], ["refreshed", "refreshed"])
        plan.apply()
        # Refreshed patches must apply again with zero fuzz/offset, in the same order.
        plan = updater.Plan(self.root)
        patch_dawn.refresh_patches(plan, source, "external.cmake", "patch")
        self.assertFalse(plan.changes)
        self.assertEqual([p["status"] for p in plan.report["patches"]], ["clean", "clean"])

    def test_conflict_blocks_roll_and_does_not_edit_checkout(self):
        text = "--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new\n"
        self.write("cmake/patches/dawn/one.patch", text)
        self.write(
            "external.cmake",
            "set(ONNXRUNTIME_Dawn_PATCH_COMMAND\n${PROJECT_SOURCE_DIR}/patches/dawn/one.patch)\nonnxruntime_fetchcontent_declare(",
        )
        plan = updater.Plan(self.root)
        patch_dawn.refresh_patches(plan, FakeSource(files={"file": "conflict\n"}), "external.cmake", "patch")
        self.assertTrue(plan.report["blockers"])
        self.assertFalse(plan.changes)
        with self.assertRaises(ValueError):
            plan.apply()

    def test_patch_paths_cannot_escape_temporary_directory(self):
        with self.assertRaises(ValueError):
            patch_dawn.patch_paths("--- a/../../escape\n+++ b/../../escape\n")


class CommandTests(TempTest):
    def setUp(self):
        super().setUp()
        self.write(
            updater.DEPS_FILE,
            f"other;keep;unchanged\ndawn;https://github.com/google/dawn/archive/{OLD}.zip;{'0' * 40}\n",
        )
        self.write(updater.MANIFEST, json.dumps(manifest(), indent=2) + "\n")
        self.write(updater.SWIFTSHADER_DOCKERFILE, f"ARG SWIFTSHADER_COMMIT={OLD}\n")
        self.write(updater.EXTERNAL_CMAKE, "if (FALSE)\n  # inactive\n  find_package(dawn REQUIRED)\nendif()\n")
        self.source = FakeSource(cache=self.root)
        self.contexts.enter_context(
            patch.object(
                updater,
                "archive_for_revision",
                return_value=(self.source, f"https://github.com/google/dawn/archive/{NEW}.zip", "4" * 40, None),
            )
        )
        self.contexts.enter_context(patch.object(updater, "Source", return_value=self.source))
        self.refresh_patches = self.contexts.enter_context(patch.object(patch_dawn, "refresh_patches"))
        self.contexts.enter_context(redirect_stdout(io.StringIO()))
        self.contexts.enter_context(redirect_stderr(io.StringIO()))
        self.args = ["--repo-root", str(self.root), "--revision", NEW]

    def test_plan_apply_check_end_to_end_and_json_report(self):
        before = (self.root / updater.DEPS_FILE).read_bytes()
        self.assertEqual(updater.main(self.args), 0)
        self.assertEqual((self.root / updater.DEPS_FILE).read_bytes(), before)
        report_text = (self.root / "build/dawn-upgrade/report.md").read_text()
        self.assertIn(updater.DEPS_FILE, report_text)
        self.assertIn("### Changed (0)", report_text)
        self.assertIn("it does not certify a build", report_text)
        self.assertEqual(updater.main([*self.args, "--check"]), 1)
        self.assertEqual(updater.main([*self.args, "--apply"]), 0)
        self.assertEqual(updater.main([*self.args, "--check"]), 0)
        self.assertIn("other;keep;unchanged", (self.root / updater.DEPS_FILE).read_text())
        report = json.loads((self.root / "build/dawn-upgrade/report.json").read_text())
        self.assertEqual(report["commit"], NEW)
        self.assertEqual(report["changedFiles"], [])

    def test_blocker_leaves_inputs_intact_and_produces_report(self):
        self.source.files[updater.FETCHER] = "required_submodules = ['unknown']"
        before = (self.root / updater.DEPS_FILE).read_bytes()
        self.assertEqual(updater.main([*self.args, "--apply"]), 2)
        self.assertEqual((self.root / updater.DEPS_FILE).read_bytes(), before)
        self.assertTrue(json.loads((self.root / "build/dawn-upgrade/report.json").read_text())["blockers"])
        self.assertIn("unknown", (self.root / "build/dawn-upgrade/report.md").read_text())

    def test_roll_does_not_read_or_change_runtime_packaging(self):
        self.write("setup.py", 'runtime_libraries = ["dxil.dll", "dxcompiler.dll"]\n')
        before = (self.root / "setup.py").read_bytes()
        self.assertEqual(updater.main([*self.args, "--apply"]), 0)
        self.assertEqual((self.root / "setup.py").read_bytes(), before)
        report = json.loads((self.root / "build/dawn-upgrade/report.json").read_text())
        self.assertEqual(
            set(report["changedFiles"]), {updater.DEPS_FILE, updater.MANIFEST, updater.SWIFTSHADER_DOCKERFILE}
        )

    def test_invalid_patch_metadata_is_reported_without_writes(self):
        self.refresh_patches.side_effect = ValueError("upstream removed a patched file")
        before = (self.root / updater.DEPS_FILE).read_bytes()
        self.assertEqual(updater.main([*self.args, "--apply"]), 2)
        self.assertEqual((self.root / updater.DEPS_FILE).read_bytes(), before)
        self.assertIn("upstream removed a patched file", (self.root / "build/dawn-upgrade/report.md").read_text())

    def test_report_distinguishes_stale_manifest_from_current_pin(self):
        original = manifest()
        original["registrations"][0]["component"]["git"]["commitHash"] = CHILD
        self.write(updater.MANIFEST, json.dumps(original))
        self.assertEqual(updater.main(self.args), 0)
        report = (self.root / "build/dawn-upgrade/report.md").read_text()
        self.assertIn(f"manifest root `{CHILD}` differs from the current pin", report)

    def test_reenabled_dawn_vcpkg_path_requires_review(self):
        self.write(updater.EXTERNAL_CMAKE, "find_package(dawn REQUIRED)\n")
        before = (self.root / updater.DEPS_FILE).read_bytes()
        self.assertEqual(updater.main([*self.args, "--apply"]), 2)
        self.assertEqual((self.root / updater.DEPS_FILE).read_bytes(), before)
        report = json.loads((self.root / "build/dawn-upgrade/report.json").read_text())
        self.assertIn("vcpkg integration changed", report["blockers"][0])


if __name__ == "__main__":
    unittest.main()
