#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Update WebGPU manifest Git commits from the Dawn version pinned in cmake/deps.txt."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import validate_webgpu_cgmanifest as validator


def _assignment(tree: ast.Module, name: str) -> ast.expr:
    values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]
    if len(values) != 1:
        raise ValueError(f"expected one top-level {name} assignment in Dawn DEPS")
    return values[0]


def _dictionary(node: ast.expr) -> dict[str, ast.expr]:
    if not isinstance(node, ast.Dict):
        raise ValueError("expected a literal dictionary in Dawn DEPS")
    result = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str) or key.value in result:
            raise ValueError("expected unique literal string keys in Dawn DEPS")
        result[key.value] = value
    return result


def _string(node: ast.expr, variables: dict[str, Any]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _string(node.left, variables) + _string(node.right, variables)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Var"
        and len(node.args) == 1
        and not node.keywords
    ):
        value = variables[_string(node.args[0], {})]
        if isinstance(value, str):
            return value
    raise ValueError(f"unsupported Dawn DEPS string expression: {ast.dump(node)}")


def parse_dawn_deps(source: str) -> dict[str, dict[str, str]]:
    """Read Git pins without executing DEPS, hooks, or dependency conditions."""
    tree = ast.parse(source)
    variables = {}
    for name, value in _dictionary(_assignment(tree, "vars")).items():
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Str"
            and len(value.args) == 1
            and not value.keywords
        ):
            variables[name] = str(ast.literal_eval(value.args[0]))
        else:
            variables[name] = ast.literal_eval(value)

    dependencies = {}
    for path, entry in _dictionary(_assignment(tree, "deps")).items():
        node = entry
        condition = ""
        if isinstance(node, ast.Dict):
            fields = _dictionary(node)
            if "url" not in fields:
                continue  # CIPD/GCS packages do not identify Git source commits.
            node = fields["url"]
            if "condition" in fields:
                condition = _string(fields["condition"], variables)
        if isinstance(node, ast.Constant) and node.value is None:
            continue
        url = _string(node, variables).format(**variables)
        repository, separator, commit = url.rpartition("@")
        if not separator or not repository.startswith("https://") or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise ValueError(f"expected an HTTPS Git URL and full commit hash for {path}: {url}")
        dependencies[path] = {"repositoryUrl": repository, "commitHash": commit.lower(), "condition": condition}
    if not dependencies:
        raise ValueError("no Git dependencies found in Dawn DEPS")
    return dependencies


def _resolve_dawn_commit(pin_field: str, pin: str) -> str:
    if pin_field == "commitHash":
        return pin
    ref = f"refs/tags/{pin}"
    result = subprocess.run(
        ["git", "ls-remote", validator.DAWN_REPOSITORY_URL, ref, ref + "^{}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    refs = dict(line.split()[::-1] for line in result.stdout.splitlines())
    # Annotated tags point at tag objects; use the peeled commit when present.
    commit = refs.get(ref + "^{}", refs.get(ref, ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"could not resolve Dawn tag {pin} to a commit")
    return commit


def _download_deps(commit: str) -> str:
    url = f"https://raw.githubusercontent.com/google/dawn/{commit}/DEPS"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _repository_identity(url: str) -> str:
    url = url.rstrip("/").removesuffix(".git")
    mirror = "https://chromium.googlesource.com/external/github.com/"
    # LLVM split repositories have no equivalent GitHub owner/repo identity.
    if url.startswith(mirror) and len(url.removeprefix(mirror).split("/")) == 2:
        url = "https://github.com/" + url.removeprefix(mirror)
    url = url.replace(
        "https://chromium.googlesource.com/external/gitlab.freedesktop.org/", "https://gitlab.freedesktop.org/"
    )
    if url.startswith("https://github.com/"):
        url = url.lower()
    return url


def update_manifest(
    manifest: dict[str, Any], commit: str, tag: str | None, dependencies: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Refresh registered Dawn pins, preserving classifications and unrelated entries."""
    updated = copy.deepcopy(manifest)
    dawn = validator._find_git_registration(updated, validator.DAWN_REPOSITORY_URL)
    dawn_git = dawn["component"]["git"]
    dawn_git["commitHash"] = commit
    if tag is None:
        dawn_git.pop("tag", None)
    else:
        dawn_git["tag"] = tag

    errors = []
    for registration in updated["registrations"]:
        comments = registration.get("comments", "")
        if "Dawn DEPS" not in comments:
            continue
        git = validator._git_component(registration)
        path_match = re.search(r"\bDawn DEPS ([^\s,;]+)", comments)
        if git is None or path_match is None or "tag" in git:
            errors.append(f"expected an untagged Git registration with a Dawn DEPS path: {comments}")
            continue
        old_path = path_match.group(1).rstrip(".")
        identity = _repository_identity(git["repositoryUrl"])
        candidates = [p for p, dep in dependencies.items() if _repository_identity(dep["repositoryUrl"]) == identity]
        if old_path in dependencies:
            candidates = [old_path] if old_path in candidates else []
        if len(candidates) != 1:
            errors.append(
                f"{old_path}: cannot uniquely match {git['repositoryUrl']}; review removed/changed repositories"
            )
            continue
        path = candidates[0]
        registration["component"]["git"]["commitHash"] = dependencies[path]["commitHash"]
        registration["comments"] = comments.replace(f"Dawn DEPS {old_path}", f"Dawn DEPS {path}", 1)
        roots = registration.setdefault(
            "dependencyRoots", [{"type": "git", "git": {"repositoryUrl": validator.DAWN_REPOSITORY_URL}}]
        )
        if len(roots) != 1 or roots[0].get("git", {}).get("repositoryUrl") != validator.DAWN_REPOSITORY_URL:
            errors.append(f"{old_path}: expected one Dawn dependency root")
            continue
        roots[0]["git"]["commitHash"] = commit
        if "tag" in roots[0]["git"]:
            if tag is None:
                roots[0]["git"].pop("tag")
            else:
                roots[0]["git"]["tag"] = tag

    if errors:
        raise ValueError(
            "\n".join(errors) + "\nNo manifest written. Use --list-deps to inspect the pinned dependencies."
        )
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=validator.WEBGPU_CGMANIFEST)
    parser.add_argument("--deps-file", type=Path, default=validator.DEPS_TXT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="check registered Dawn hashes without writing")
    modes.add_argument("--list-deps", action="store_true", help="print all pinned Dawn Git dependencies as JSON")
    args = parser.parse_args(argv)
    try:
        pin_field, pin = validator._dawn_pin_from_deps_txt(args.deps_file)
        commit = _resolve_dawn_commit(pin_field, pin)
        dependencies = parse_dawn_deps(_download_deps(commit))
        print(f"Dawn {pin} -> {commit}; {len(dependencies)} Git dependencies", file=sys.stderr)
        if args.list_deps:
            print(json.dumps(dependencies, indent=2) + "\n", end="")
            return 0
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        updated = update_manifest(manifest, commit, pin if pin_field == "tag" else None, dependencies)
        if updated == manifest:
            print(f"Registered Dawn hashes are up to date in {args.manifest}")
            return 0
        if args.check:
            print(
                f"ERROR: stale Dawn hashes or paths in {args.manifest}; run without --check to update", file=sys.stderr
            )
            return 1
        # Resolve and validate every entry before touching the maintained manifest.
        args.manifest.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {args.manifest}. Review dependency additions/removals and classifications with --list-deps.")
    except (OSError, ValueError, KeyError, SyntaxError, subprocess.SubprocessError) as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
