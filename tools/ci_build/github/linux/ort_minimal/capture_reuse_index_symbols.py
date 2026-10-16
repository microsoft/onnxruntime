#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Temporary, read-only-build Android reuse-index link attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess

from readelf_utils import get_section_sizes

SOURCE = "onnxruntime/core/framework/allocation_planner.cc"
REPO = pathlib.Path(__file__).resolve().parents[5]


def run(command, cwd, output=None):
    if command[0] == "git":
        command = ["git", "-c", f"safe.directory={REPO}", *command[1:]]
    result = subprocess.run(command, cwd=cwd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if output:
        output.write_text(result.stdout)
    if result.returncode:
        print(result.stdout, flush=True)
    result.check_returncode()
    return result.stdout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand_responses(command, directory):
    expanded = []
    for token in command:
        if token.startswith("@"):
            expanded.extend(expand_responses(shlex.split((directory / token[1:]).read_text()), directory))
        else:
            expanded.append(token)
    return expanded


def link_command(command, output, map_path, replacement_archive=None):
    command = [token for token in command if token not in ("-s", "-Wl,-s")]
    command[command.index("-o") + 1] = str(output)
    if replacement_archive:
        indices = [i for i, token in enumerate(command) if pathlib.Path(token).name == "libonnxruntime_framework.a"]
        if len(indices) != 1:
            raise ValueError("Expected exactly one framework archive in the expanded link command")
        command[indices[0]] = str(replacement_archive)
    return [*command, f"-Wl,-Map={map_path}"]


def compile_command(command, source, output, original_source):
    command = list(command)
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(output)
    for flag, value in (("-MF", str(output) + ".d"), ("-MT", str(output))):
        if flag in command:
            command[command.index(flag) + 1] = value
    # Preserve __FILE__/source-location strings despite compiling the base in a separate directory.
    command.append(f"-ffile-prefix-map={source}={original_source}")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--candidate-source-sha256", required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    compiler_work = out / "compiler-work"
    compiler_work.mkdir()
    os.environ["TMPDIR"] = str(compiler_work)

    if digest(REPO / SOURCE) != args.candidate_source_sha256:
        raise ValueError("Unexpected candidate source; refusing a mislabeled comparison")
    shutil.copyfile(build / "CMakeCache.txt", out / "CMakeCache.txt")
    run(["git", "fetch", "--no-tags", "--depth=1", "origin", args.base_ref], REPO, out / "fetch.log")
    changed = run(["git", "diff", "--name-only", args.base_ref, "HEAD", "--", "include", "onnxruntime", "cmake"], REPO)
    allowed = {SOURCE, "onnxruntime/test/framework/allocation_planner_test.cc"}
    if set(changed.splitlines()) - allowed:
        raise ValueError(f"Base/candidate runtime inputs differ beyond the planner: {changed}")

    link_file = build / "CMakeFiles/onnxruntime.dir/link.txt"
    original_link = expand_responses(shlex.split(link_file.read_text()), build)
    if any(token in ("&&", ";", "|") for token in original_link):
        raise ValueError("Expected one compiler-driver link command, not a shell program")
    tools = pathlib.Path(original_link[0]).parent
    archive_tokens = [token for token in original_link if pathlib.Path(token).name == "libonnxruntime_framework.a"]
    if len(archive_tokens) != 1:
        raise ValueError("Expected one framework archive")
    archive = (build / archive_tokens[0]).resolve()
    normal = (build / original_link[original_link.index("-o") + 1]).resolve()
    original_hashes = {str(path): digest(path) for path in (archive, normal)}
    run([str(tools / "clang++"), "--version"], build, out / "compiler-version.txt")

    make_file = build / "CMakeFiles/onnxruntime_framework.dir/build.make"
    targets = {
        line.split(":")[0]
        for line in make_file.read_text().splitlines()
        if line.startswith("CMakeFiles/") and "/allocation_planner.cc.o:" in line
    }
    if len(targets) != 1:
        raise ValueError(f"Expected one generated planner object target: {targets}")
    dry_run = run(
        ["make", "--dry-run", "--always-make", "-f", str(make_file), next(iter(targets))],
        build,
        out / "compile-dry-run.txt",
    )
    commands = [shlex.split(line) for line in dry_run.splitlines() if " -c " in line and SOURCE in line]
    if len(commands) != 1 or "-c" not in commands[0]:
        raise ValueError("Expected exactly one planner compilation in the generated dry run")
    original_compile = commands[0]
    original_source = original_compile[original_compile.index("-c") + 1]

    baseline = out / "base"
    baseline.mkdir()
    baseline_source = baseline / "allocation_planner.cc"
    baseline_source.write_text(run(["git", "show", f"{args.base_ref}:{SOURCE}"], REPO))
    baseline_object = baseline / "allocation_planner.cc.o"
    baseline_compile = compile_command(original_compile, baseline_source, baseline_object, original_source)
    run(baseline_compile, build, baseline / "compile.log")
    baseline_archive = baseline / archive.name
    shutil.copyfile(archive, baseline_archive)
    members = run([str(tools / "llvm-ar"), "t", str(archive)], build).splitlines()
    if members.count(baseline_object.name) != 1:
        raise ValueError("Expected exactly one planner archive member")
    run([str(tools / "llvm-ar"), "rcs", str(baseline_archive), str(baseline_object)], build, baseline / "archive.log")
    if run([str(tools / "llvm-ar"), "t", str(baseline_archive)], build).splitlines() != members:
        raise ValueError("Baseline archive member ordering changed")

    report = {
        "kind": "targeted-baseline-TU-and-existing-CI-archives-relink-NOT-full-base-build",
        "head": run(["git", "rev-parse", "HEAD"], REPO).strip(),
        "base": args.base_ref,
        "candidate_source_sha256": digest(REPO / SOURCE),
        "base_source_sha256": digest(baseline_source),
        "original_compile": original_compile,
        "base_compile": baseline_compile,
        "original_link": original_link,
        "original_library_sections": get_section_sizes(normal, str(tools / "llvm-readelf")),
        "variants": {},
    }
    for variant, replacement in (("candidate", None), ("base", baseline_archive)):
        directory = out / variant
        directory.mkdir(exist_ok=True)
        library = directory / "libonnxruntime.unstripped.so"
        command = link_command(original_link, library, directory / "link.map", replacement)
        run(command, build, directory / "link.log")
        run(
            [str(tools / "llvm-nm"), "--print-size", "--size-sort", "--defined-only", "--demangle", str(library)],
            build,
            directory / "symbols.txt",
        )
        stripped = directory / "libonnxruntime.stripped.so"
        run([str(tools / "llvm-strip"), "--strip-all", "-o", str(stripped), str(library)], build)
        sections = get_section_sizes(stripped, str(tools / "llvm-readelf"))
        report["variants"][variant] = {"link": command, "sections": sections, "total": sum(sections.values())}
    report["candidate_relink_sections_exact"] = (
        report["variants"]["candidate"]["sections"] == report["original_library_sections"]
    )
    report["original_artifacts_unchanged"] = all(
        digest(pathlib.Path(path)) == sha for path, sha in original_hashes.items()
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    baseline_archive.unlink()
    shutil.rmtree(compiler_work)
    print(json.dumps(report, indent=2))
    if not report["candidate_relink_sections_exact"] or not report["original_artifacts_unchanged"]:
        raise ValueError("Diagnostic relink changed original code sections or build artifacts")


if __name__ == "__main__":
    main()
