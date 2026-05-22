#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Validate WebGPU Component Governance manifest drift."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBGPU_CGMANIFEST = Path(__file__).resolve().with_name("cgmanifest.webgpu.json")
DEPS_TXT = REPO_ROOT / "cmake" / "deps.txt"
PLUGIN_WIN_WEBGPU_STAGE = (
    REPO_ROOT / "tools" / "ci_build" / "github" / "azure-pipelines" / "stages" / "plugin-win-webgpu-stage.yml"
)

DAWN_REPOSITORY_URL = "https://github.com/google/dawn.git"
DXC_REPOSITORY_URL = "https://github.com/microsoft/DirectXShaderCompiler.git"


def _load_manifest() -> dict[str, Any]:
    with WEBGPU_CGMANIFEST.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    registrations = manifest.get("registrations")
    if not isinstance(registrations, list):
        raise ValueError(f"{WEBGPU_CGMANIFEST} must contain a registrations array")

    return manifest


def _git_component(registration: dict[str, Any]) -> dict[str, str] | None:
    component = registration.get("component")
    if not isinstance(component, dict) or component.get("type") != "git":
        return None

    git = component.get("git")
    if not isinstance(git, dict):
        return None

    repository_url = git.get("repositoryUrl")
    commit_hash = git.get("commitHash")
    if not isinstance(repository_url, str) or not isinstance(commit_hash, str):
        return None

    result = {"repositoryUrl": repository_url, "commitHash": commit_hash}
    tag = git.get("tag")
    if isinstance(tag, str):
        result["tag"] = tag
    return result


def _registrations(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return manifest["registrations"]


def _find_git_registration(manifest: dict[str, Any], repository_url: str, *, tag: str | None = None) -> dict[str, Any]:
    matches = []
    for registration in _registrations(manifest):
        git = _git_component(registration)
        if git is None or git["repositoryUrl"] != repository_url:
            continue
        if tag is not None and git.get("tag") != tag:
            continue
        matches.append(registration)

    if len(matches) != 1:
        suffix = f" with tag {tag}" if tag is not None else ""
        raise ValueError(f"expected exactly one registration for {repository_url}{suffix}, found {len(matches)}")
    return matches[0]


def _dawn_commit_from_deps_txt() -> str:
    deps_text = DEPS_TXT.read_text(encoding="utf-8")
    match = re.search(r"^dawn;https://github\.com/google/dawn/archive/([0-9a-f]{40})\.zip;", deps_text, re.MULTILINE)
    if not match:
        raise ValueError(f"could not find Dawn commit in {DEPS_TXT}")
    return match.group(1)


def _dxc_release_from_pipeline() -> tuple[str, str, str]:
    pipeline_text = PLUGIN_WIN_WEBGPU_STAGE.read_text(encoding="utf-8")
    url_match = re.search(r'\$dxcZipUrl = "([^"]+)"', pipeline_text)
    hash_match = re.search(r'\$expectedHash = "([0-9A-Fa-f]+)"', pipeline_text)
    if not url_match or not hash_match:
        raise ValueError(f"could not find DXC release URL/hash in {PLUGIN_WIN_WEBGPU_STAGE}")

    tag_match = re.search(r"/download/(v[^/]+)/", url_match.group(1))
    if not tag_match:
        raise ValueError(f"could not find DXC release tag in {url_match.group(1)}")

    return tag_match.group(1), url_match.group(1), hash_match.group(1).upper()


def _validate_dawn_root(manifest: dict[str, Any]) -> None:
    registration = _find_git_registration(manifest, DAWN_REPOSITORY_URL)
    git = _git_component(registration)
    if git is None:
        raise ValueError("Dawn registration must be a git component")

    expected_commit = _dawn_commit_from_deps_txt()
    if git["commitHash"] != expected_commit:
        raise ValueError(f"Dawn manifest commit {git['commitHash']} does not match {DEPS_TXT} commit {expected_commit}")


def _validate_dxc_release(manifest: dict[str, Any]) -> None:
    expected_tag, expected_url, expected_hash = _dxc_release_from_pipeline()
    registration = _find_git_registration(manifest, DXC_REPOSITORY_URL, tag=expected_tag)
    git = _git_component(registration)
    if git is None:
        raise ValueError(f"DXC {expected_tag} registration must be a git component")

    comments = registration.get("comments", "")
    if expected_url not in comments or expected_hash not in comments:
        raise ValueError(
            f"DXC {expected_tag} registration comments must contain pipeline URL {expected_url} "
            f"and SHA256 {expected_hash}"
        )


def _validate_dawn_dependency_roots(manifest: dict[str, Any]) -> None:
    dawn_commit = _dawn_commit_from_deps_txt()

    for registration in _registrations(manifest):
        comments = registration.get("comments", "")
        if not isinstance(comments, str) or "Dawn DEPS" not in comments:
            continue

        dependency_roots = registration.get("dependencyRoots")
        if not isinstance(dependency_roots, list) or len(dependency_roots) != 1:
            raise ValueError(f"Dawn-derived registration is missing one dependencyRoots entry: {comments}")

        root = dependency_roots[0]
        if not isinstance(root, dict):
            raise ValueError(f"Dawn dependency root must be an object: {comments}")

        root_git = root.get("git")
        if root.get("type") != "git" or not isinstance(root_git, dict):
            raise ValueError(f"Dawn dependency root must be a git component: {comments}")
        if root_git.get("repositoryUrl") != DAWN_REPOSITORY_URL or root_git.get("commitHash") != dawn_commit:
            raise ValueError(f"Dawn dependency root does not match {DAWN_REPOSITORY_URL}@{dawn_commit}: {comments}")


def main() -> int:
    try:
        manifest = _load_manifest()
        _validate_dawn_root(manifest)
        _validate_dxc_release(manifest)
        _validate_dawn_dependency_roots(manifest)
    except (OSError, ValueError) as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        return 1

    print(f"Validated {WEBGPU_CGMANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
