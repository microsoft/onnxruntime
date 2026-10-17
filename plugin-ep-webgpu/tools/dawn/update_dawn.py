#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Prepare and check a Dawn roll, including archive, manifest, patches and test dependencies."""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import patch_dawn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "cgmanifests/webgpu"))
import update_webgpu_cgmanifest as cg  # noqa: E402
import validate_webgpu_cgmanifest as validator  # noqa: E402

DEPS_FILE = "cmake/deps.txt"
MANIFEST = "cgmanifests/webgpu/cgmanifest.webgpu.json"
EXTERNAL_CMAKE = "cmake/external/onnxruntime_external_deps.cmake"
SWIFTSHADER_DOCKERFILE = "tools/ci_build/github/linux/docker/inference/shared/python/webgpu/Dockerfile"
FETCHER = "tools/fetch_dawn_dependencies.py"


def digest(path: Path) -> str:
    checksum = hashlib.sha1(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class Source:
    """Read pinned upstream files; never execute DEPS or extract an entire source archive."""

    def __init__(self, commit: str, cache: Path, archive: Path | None = None):
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"expected a full Dawn commit hash: {commit}")
        self.commit = commit
        self.cache = cache
        self.archive = archive

    def read(self, path: str) -> str:
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in path:
            raise ValueError(f"invalid Dawn source path: {path}")
        if self.archive is not None:
            with zipfile.ZipFile(self.archive) as archive:
                names = [n for n in archive.namelist() if n.partition("/")[2] == path]
                if len(names) != 1:
                    raise ValueError(f"expected exactly one {path} in {self.archive}")
                return archive.read(names[0]).decode("utf-8").replace("\r\n", "\n")
        destination = self.cache / self.commit / path
        if not destination.exists():
            download(f"https://raw.githubusercontent.com/google/dawn/{self.commit}/{path}", destination)
        return destination.read_text(encoding="utf-8")


def archive_for_revision(revision: str, cache: Path) -> tuple[Source, str, str, str | None]:
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", revision) or ".." in revision:
        raise ValueError("revision must be a full commit hash or an exact Dawn tag")
    is_commit = re.fullmatch(r"[0-9a-f]{40}", revision) is not None
    commit = cg._resolve_dawn_commit("commitHash" if is_commit else "tag", revision)
    tag = None if is_commit else revision
    suffix = revision if is_commit else f"refs/tags/{revision}"
    url = f"https://github.com/google/dawn/archive/{suffix}.zip"
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    archive = cache / f"{commit}-{key}.zip"
    checksum_file = archive.with_suffix(".sha1")
    if not archive.exists() or not checksum_file.exists():
        print(f"Downloading {url}", file=sys.stderr, flush=True)
        download(url, archive)
        checksum_file.write_text(digest(archive), encoding="ascii")
    archive_hash = digest(archive)
    if checksum_file.read_text(encoding="ascii").strip() != archive_hash:
        raise ValueError(f"cached Dawn archive is corrupt; remove {archive} and retry")
    source = Source(commit, cache, archive)
    # A tag archive and metadata resolved through Git must describe the same roll.
    if source.read("DEPS") != Source(commit, cache).read("DEPS"):
        raise ValueError("Dawn archive DEPS does not match the resolved commit; remove the cached archive and retry")
    if tag is not None and cg._resolve_dawn_commit("tag", tag) != commit:
        raise ValueError(f"Dawn tag {tag} moved during the download; retry the roll")
    return source, url, archive_hash, tag


def required_paths(source: str) -> list[str]:
    tree = ast.parse(source)
    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "required_submodules" for t in node.targets)
    ]
    if len(values) != 1:
        raise ValueError("expected one required_submodules list in Dawn's dependency fetcher")
    paths = ast.literal_eval(values[0])
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise ValueError("required_submodules must be a nonempty list of paths")
    return paths


def dependency_diff(old: dict, new: dict) -> dict:
    return {
        "added": {path: new[path] for path in sorted(new.keys() - old.keys())},
        "removed": {path: old[path] for path in sorted(old.keys() - new.keys())},
        "changed": {
            path: {"old": old[path], "new": new[path]}
            for path in sorted(old.keys() & new.keys())
            if old[path] != new[path]
        },
    }


def refresh_manifest(manifest: dict, source: Source, tag: str | None, deps: dict, report: dict) -> dict:
    updated = copy.deepcopy(manifest)
    root = validator._find_git_registration(updated, validator.DAWN_REPOSITORY_URL)
    manifest_commit = root["component"]["git"]["commitHash"]
    report["previousManifestCommit"] = manifest_commit
    previous = cg.parse_dawn_deps(Source(manifest_commit, source.cache).read("DEPS"))
    identities = {cg._repository_identity(dep["repositoryUrl"]) for dep in deps.values()}
    previous_identities = {cg._repository_identity(dep["repositoryUrl"]) for dep in previous.values()}
    removed = []
    for registration in list(updated["registrations"]):
        if "Dawn DEPS" not in registration.get("comments", ""):
            continue
        git = validator._git_component(registration)
        if git is None:
            raise ValueError("Dawn-derived registration is not a Git component")
        identity = cg._repository_identity(git["repositoryUrl"])
        path_match = re.search(r"\bDawn DEPS ([^\s,;]+)", registration["comments"])
        path = path_match.group(1).rstrip(".") if path_match else ""
        if identity not in identities and path not in deps:
            # Only retire an obsolete development input that can be verified in
            # the manifest's previous Dawn version. Runtime policy needs review.
            if registration.get("developmentDependency") is True and identity in previous_identities:
                updated["registrations"].remove(registration)
                removed.append(registration)
    report["removedDevelopmentRegistrations"] = removed
    updated = cg.update_manifest(updated, source.commit, tag, deps)
    registered = {
        (cg._repository_identity(r["component"]["git"]["repositoryUrl"]), r["component"]["git"]["commitHash"])
        for r in updated["registrations"]
        if "Dawn DEPS" in r.get("comments", "")
    }
    required = required_paths(source.read(FETCHER))
    report["requiredPaths"] = required

    def is_registered(dependency: dict) -> bool:
        return (cg._repository_identity(dependency["repositoryUrl"]), dependency["commitHash"]) in registered

    missing = [p for p in required if p not in deps or not is_registered(deps[p])]
    if missing:
        report["blockers"].append(f"Review new or nested fetched dependencies and their classifications: {missing}")
    report["unregisteredGitDependencies"] = {p: d for p, d in deps.items() if not is_registered(d)}
    report["inventoryCounts"] = {
        "declaredGitDependencies": len(deps),
        "fetcherPaths": len(required),
        "registeredDawnDependencies": sum("Dawn DEPS" in r.get("comments", "") for r in updated["registrations"]),
    }
    report["conditionChanges"] = dependency_diff(previous, deps)["changed"]
    report["conditionChanges"] = {
        p: {"old": d["old"]["condition"], "new": d["new"]["condition"]}
        for p, d in report["conditionChanges"].items()
        if d["old"]["condition"] != d["new"]["condition"]
    }
    return updated


def format_report(report: dict) -> str:
    """Render the prepared edits and review decisions without implying build validation."""
    lines = [
        "# Dawn upgrade report",
        "",
        f"- Current pin: `{report['oldRevision']}` (`{report['oldCommit']}`)",
        f"- Requested pin: `{report['revision']}` (`{report['commit']}`)",
        f"- Archive: {report['archiveUrl']}",
        f"- Archive SHA1: `{report['archiveSHA1']}`",
        "",
    ]
    if report.get("previousManifestCommit", report["oldCommit"]) != report["oldCommit"]:
        lines += [
            f"The existing manifest root `{report['previousManifestCommit']}` differs from the current pin.",
            "The plan also repairs that drift using the requested revision's DEPS.",
            "",
        ]
    lines += ["## Required review", ""]
    lines += [f"- {message}" for message in report["blockers"]] or ["No metadata or patch blockers found."]
    lines += ["", "## Prepared files", ""]
    lines += [f"- `{path}`" for path in report["changedFiles"]] or ["No tracked changes needed."]
    counts = report.get("inventoryCounts")
    if counts:
        lines += [
            "",
            (
                f"Git paths in DEPS: {counts['declaredGitDependencies']}; fetcher paths: "
                f"{counts['fetcherPaths']}; reviewed Dawn registrations: {counts['registeredDawnDependencies']}. "
                "These inventories have different scopes."
            ),
        ]
    lines += ["", "## Dependency changes", ""]
    for category, changes in report["dependencyChanges"].items():
        lines += [f"### {category.capitalize()} ({len(changes)})", ""]
        for path, entry in changes.items():
            if category == "changed":
                before, after = entry["old"], entry["new"]
                lines.append(f"- `{path}`: `{before['commitHash']}` -> `{after['commitHash']}`")
                if before["repositoryUrl"] != after["repositoryUrl"]:
                    lines.append(f"  Repository: {before['repositoryUrl']} -> {after['repositoryUrl']}")
                if before["condition"] != after["condition"]:
                    lines.append(f"  Condition: `{before['condition']}` -> `{after['condition']}`")
            else:
                lines.append(f"- `{path}`: {entry['repositoryUrl']} at `{entry['commitHash']}`")
        if not changes:
            lines.append("None.")
        lines.append("")
    lines += ["## Patch checks", "", "| Patch | Result | Context fuzz |", "| --- | --- | --- |"]
    for patch in report.get("patches", []):
        lines.append(f"| `{patch['path']}` | {patch['status']} | {patch.get('contextFuzz', False)} |")
    lines += ["", "Review refreshed patch context in `changes.patch`; detailed patch output is in `report.json`."]
    lines += ["", "## Validation still required", ""]
    lines += [f"- {note}" for note in report["notes"]]
    lines += ["", "This report prepares source edits; it does not certify a build, package, or GPU run.", ""]
    return "\n".join(lines)


class Plan:
    def __init__(self, root: Path):
        self.root = root
        self.original: dict[str, bytes] = {}
        self.changes: dict[str, str] = {}
        self.report: dict[str, Any] = {"blockers": [], "notes": []}

    def read(self, path: str) -> str:
        if path not in self.original:
            self.original[path] = (self.root / path).read_bytes()
        return self.changes.get(path, self.original[path].decode("utf-8").replace("\r\n", "\n"))

    def edit(self, path: str, text: str) -> None:
        self.read(path)
        if text == self.original[path].decode("utf-8").replace("\r\n", "\n"):
            self.changes.pop(path, None)
        else:
            self.changes[path] = text

    def diff(self) -> str:
        return "".join(
            "".join(
                difflib.unified_diff(
                    self.original[p].decode("utf-8").replace("\r\n", "\n").splitlines(keepends=True),
                    self.changes[p].splitlines(keepends=True),
                    fromfile=f"a/{p}",
                    tofile=f"b/{p}",
                )
            )
            for p in sorted(self.changes)
        )

    def apply(self) -> None:
        if self.report["blockers"]:
            raise ValueError("cannot apply a roll with unresolved blockers; see report.json")
        # Check all inputs before the first write, and roll back completed writes
        # if a filesystem error interrupts application. Preserve local newlines.
        for path, original in self.original.items():
            if (self.root / path).read_bytes() != original:
                raise ValueError(f"{path} changed while preparing the roll; rerun")
        written = []
        try:
            for path, text in self.changes.items():
                data = text.replace("\n", "\r\n") if b"\r\n" in self.original[path] else text
                written.append(path)
                (self.root / path).write_bytes(data.encode("utf-8"))
        except OSError:
            for path in reversed(written):
                (self.root / path).write_bytes(self.original[path])
            raise


def make_plan(root: Path, revision: str | None, cache: Path, patch_executable: str) -> Plan:
    plan = Plan(root)
    old_field, old_pin = validator._dawn_pin_from_deps_txt(root / DEPS_FILE)
    old_commit = cg._resolve_dawn_commit(old_field, old_pin)
    source, url, archive_hash, tag = archive_for_revision(revision or old_pin, cache)
    deps = cg.parse_dawn_deps(source.read("DEPS"))
    previous = cg.parse_dawn_deps(Source(old_commit, cache).read("DEPS"))
    plan.report.update(
        oldRevision=old_pin,
        oldCommit=old_commit,
        revision=revision or old_pin,
        commit=source.commit,
        archiveUrl=url,
        archiveSHA1=archive_hash,
        dependencyChanges=dependency_diff(previous, deps),
    )
    deps_text, count = re.subn(
        r"^dawn;[^\r\n]+", f"dawn;{url};{archive_hash}", plan.read(DEPS_FILE), flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError("expected exactly one Dawn archive entry in cmake/deps.txt")
    plan.edit(DEPS_FILE, deps_text)
    manifest = json.loads(plan.read(MANIFEST))
    try:
        manifest = refresh_manifest(manifest, source, tag, deps, plan.report)
        plan.edit(MANIFEST, json.dumps(manifest, indent=2) + "\n")
    except ValueError as error:
        plan.report["blockers"].append(str(error))
    swiftshader = deps.get("third_party/swiftshader")
    if swiftshader is None:
        plan.report["blockers"].append("Dawn no longer declares third_party/swiftshader; review the Linux test image")
    else:
        text, count = re.subn(
            r"^ARG SWIFTSHADER_COMMIT=[0-9a-f]{40}$",
            f"ARG SWIFTSHADER_COMMIT={swiftshader['commitHash']}",
            plan.read(SWIFTSHADER_DOCKERFILE),
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError("expected one SwiftShader commit in the WebGPU Dockerfile")
        plan.edit(SWIFTSHADER_DOCKERFILE, text)

    try:
        patch_dawn.refresh_patches(plan, source, EXTERNAL_CMAKE, patch_executable)
    except ValueError as error:
        plan.report["blockers"].append(f"Review the Dawn patch stack: {error}")
    if not re.search(r"if\s*\(FALSE\)\s*(?:#[^\n]*\n\s*)*find_package\(dawn REQUIRED\)", plan.read(EXTERNAL_CMAKE)):
        plan.report["blockers"].append(
            "The Dawn vcpkg integration changed; review its pin, archive checksum and patch stack before rolling."
        )
    plan.report["notes"].extend(
        [
            "The dormant Dawn vcpkg overlay is not upgraded; its pin and patches need a separate review before re-enabling.",
            "DXC release and Agility SDK pins are independent; they were not upgraded.",
            "Run native, plugin, external-Dawn, browser, and final-package tests; this tool does not compile or run GPU tests.",
            "Review dependency condition changes and package classification; DEPS is not the shipped dependency set.",
        ]
    )
    plan.report["changedFiles"] = sorted(plan.changes)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="exact Dawn tag or full Git commit; defaults to the current pin")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="ORT checkout to inspect/update")
    parser.add_argument("--cache-dir", type=Path, help="download cache; defaults to <repo>/build/dawn-upgrade/cache")
    parser.add_argument("--output-dir", type=Path, help="report.md, report.json and changes.patch destination")
    parser.add_argument("--patch-executable", default="patch", help="GNU patch executable")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true", help="apply prepared changes; default is a read-only plan")
    modes.add_argument("--check", action="store_true", help="fail if tracked files need changes")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    output = args.output_dir or root / "build/dawn-upgrade"
    cache = args.cache_dir or root / "build/dawn-upgrade/cache"
    try:
        plan = make_plan(root, args.revision, cache, args.patch_executable)
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(json.dumps(plan.report, indent=2) + "\n", encoding="utf-8")
        (output / "report.md").write_text(format_report(plan.report), encoding="utf-8")
        (output / "changes.patch").write_text(plan.diff(), encoding="utf-8")
        print(f"Dawn: {plan.report['oldRevision']} -> {plan.report['revision']} ({plan.report['commit']})")
        print(
            f"{len(plan.changes)} changed files; {len(plan.report['blockers'])} blockers. Report: {output / 'report.md'}"
        )
        for blocker in plan.report["blockers"]:
            print(f"REVIEW: {blocker}", file=sys.stderr)
        if plan.report["blockers"]:
            return 2
        if args.apply:
            plan.apply()
            print("Applied local changes. Build and package validation are still required.")
        if args.check and plan.changes:
            return 1
        return 0
    except (OSError, ValueError, KeyError, SyntaxError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
