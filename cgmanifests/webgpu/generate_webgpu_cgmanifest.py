#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate a WebGPU build-input inventory from Dawn's fetched dependencies."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assignment(tree: ast.AST, name: str, default: ast.expr | None = None) -> ast.expr:
    matches = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]
    if not matches and default is not None:
        return default
    if len(matches) != 1:
        raise ValueError(f"Expected one literal {name} assignment in Dawn metadata; found {len(matches)}")
    return matches[0]


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
        return str(variables[ast.literal_eval(node.args[0])])
    raise ValueError(f"Unsupported Dawn dependency URL expression: {ast.dump(node)}")


def _dependency_urls(source: Path) -> dict[str, str]:
    tree = ast.parse((source / "DEPS").read_text(encoding="utf-8"))
    variable_nodes = _assignment(tree, "vars", ast.Dict(keys=[], values=[]))
    if not isinstance(variable_nodes, ast.Dict):
        raise ValueError("Expected a literal Dawn vars dictionary")
    variables = {}
    for key, value in zip(variable_nodes.keys, variable_nodes.values, strict=True):
        # Chromium's DEPS also uses Str() to keep version strings as strings.
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Str"
            and len(value.args) == 1
            and not value.keywords
        ):
            variables[ast.literal_eval(key)] = str(ast.literal_eval(value.args[0]))
        else:
            variables[ast.literal_eval(key)] = ast.literal_eval(value)
    deps = _assignment(tree, "deps")
    if not isinstance(deps, ast.Dict):
        raise ValueError("Expected a literal Dawn DEPS dictionary")
    urls = {}
    for key, value in zip(deps.keys, deps.values, strict=True):
        if not isinstance(value, ast.Dict):
            continue
        for field, expr in zip(value.keys, value.values, strict=True):
            if isinstance(field, ast.Constant) and field.value == "url":
                urls[ast.literal_eval(key)] = _string(expr, variables).format(**variables)
    return urls


def _git_component(source: Path, url: str) -> dict[str, Any]:
    repository, pin = url.rsplit("@", 1)
    if not re.fullmatch("[0-9a-fA-F]{40}", pin):
        raise ValueError(f"Expected a commit pin for {source}")
    if not (source / ".git").exists():
        raise ValueError(f"Missing fetched dependency: {source}. Run the Dawn configure/fetch step first.")
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    actual = result.stdout.strip()
    if actual.lower() != pin.lower():
        raise ValueError(f"Stale dependency {source}: expected {pin}, found {actual}")
    # Use public identities even when CI redirects fetches to authenticated mirrors.
    repository = repository.replace("https://chromium.googlesource.com/external/github.com/", "https://github.com/")
    return {"type": "git", "git": {"repositoryUrl": repository, "commitHash": actual}}


def _collect(source: Path, paths: list[str]) -> list[dict[str, Any]]:
    urls = _dependency_urls(source)
    components = []
    for path in paths:
        if path not in urls:
            continue  # Dawn's fetcher also skips entries absent from this DEPS file.
        checkout = (source / path).resolve()
        if not checkout.is_relative_to(source.resolve()):
            raise ValueError(f"Dependency escapes its source directory: {path}")
        components.append(_git_component(checkout, urls[path]))
        nested = [p[len(path) + 1 :] for p in paths if p.startswith(path + "/")]
        if nested and (checkout / "DEPS").is_file():
            components.extend(_collect(checkout, nested))
    return components


def _archive(name: str, version: str, url: str, digest: str) -> dict[str, Any]:
    return {
        "type": "other",
        "other": {"name": name, "version": version, "downloadUrl": url, "hash": digest},
    }


def generate(dawn_source: Path, deps_file: Path, dxc_pipeline: Path | None = None) -> dict[str, Any]:
    entries = [
        line.split(";") for line in deps_file.read_text(encoding="utf-8").splitlines() if line.startswith("dawn;")
    ]
    if len(entries) != 1 or len(entries[0]) != 3:
        raise ValueError(f"Expected one Dawn archive pin in {deps_file}")
    _, url, digest = entries[0]
    if not re.fullmatch("[0-9a-fA-F]{40}", digest):
        raise ValueError("Expected the Dawn archive SHA1 from cmake/deps.txt")
    version = url.rsplit("/", 1)[-1].removesuffix(".zip")
    # FetchContent extracts an archive without Dawn's git metadata. Record the
    # verified archive identity instead of resolving a tag again over the network.
    dawn = _archive("Dawn", version, url, digest)
    fetcher = ast.parse((dawn_source / "tools/fetch_dawn_dependencies.py").read_text(encoding="utf-8"))
    paths = ast.literal_eval(_assignment(fetcher, "required_submodules"))
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise ValueError("Expected Dawn's required_submodules to be a nonempty list of paths")
    components = _collect(dawn_source, paths)
    if not components:
        raise ValueError("No fetched Dawn dependencies were found")
    registrations = [{"component": dawn}]
    unique = {json.dumps(component, sort_keys=True): component for component in components}
    for key in sorted(unique):
        registrations.append({"component": unique[key], "dependencyRoots": [dawn]})
    if dxc_pipeline is not None:
        text = dxc_pipeline.read_text(encoding="utf-8")
        url_match = re.search(r'\$dxcZipUrl = "([^"]+)"', text)
        hash_match = re.search(r'\$expectedHash = "([0-9a-fA-F]{64})"', text)
        if not url_match or not hash_match:
            raise ValueError(f"Cannot find the redistributed DXC archive URL/SHA256 in {dxc_pipeline}")
        dxc_url = url_match.group(1)
        tag = re.search(r"/releases/download/([^/]+)/", dxc_url)
        if tag is None:
            raise ValueError(f"Cannot find the DXC release version in {dxc_pipeline}")
        registrations.append(
            {"component": _archive("DirectXShaderCompiler", tag.group(1), dxc_url, hash_match.group(1))}
        )
    return {"version": 1, "registrations": registrations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dawn-source", required=True, type=Path, help="Dawn source used by a completed configure step"
    )
    parser.add_argument("--deps-file", type=Path, default=REPO_ROOT / "cmake/deps.txt")
    parser.add_argument("--dxc-pipeline", type=Path, help="Windows packaging pipeline that supplies the DXC release")
    parser.add_argument("--output", type=Path, action="append", required=True, help="Manifest destination; repeatable")
    args = parser.parse_args()
    try:
        manifest = generate(args.dawn_source, args.deps_file, args.dxc_pipeline)
        contents = json.dumps(manifest, indent=2) + "\n"
        for output in args.output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(contents, encoding="utf-8")
        print(f"Generated {len(manifest['registrations'])} WebGPU component registrations")
    except (OSError, ValueError, KeyError, SyntaxError, subprocess.CalledProcessError) as ex:
        parser.exit(1, f"ERROR: {ex}\n")


if __name__ == "__main__":
    main()
