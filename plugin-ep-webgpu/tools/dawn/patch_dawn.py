# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Check and refresh ORT's active Dawn patches in their build order."""

import difflib
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


def patch_paths(text: str) -> list[str]:
    before = re.findall(r"^--- a/(.+)$", text, re.MULTILINE)
    after = re.findall(r"^\+\+\+ b/(.+)$", text, re.MULTILINE)
    if not before or before != after:
        raise ValueError("automatic patch refresh supports modifications of existing files only")
    for path in before:
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or "\\" in path:
            raise ValueError(f"unsafe patch path: {path}")
    return before


def refresh_patches(plan, source, cmake_path: str, executable: str) -> None:
    cmake = plan.read(cmake_path)
    block = re.search(
        r"set\(ONNXRUNTIME_Dawn_PATCH_COMMAND\s+(.*?)\n\s*onnxruntime_fetchcontent_declare", cmake, re.DOTALL
    )
    if block is None:
        raise ValueError("cannot locate the active Dawn patch command in CMake")
    names = re.findall(r"\$\{PROJECT_SOURCE_DIR\}/(patches/dawn/[\w.-]+\.patch)", block.group(1))
    if not names or len(names) != len(set(names)):
        raise ValueError("expected a nonempty, unique, ordered Dawn patch list")
    results = []
    all_touched = set()
    with tempfile.TemporaryDirectory(prefix="ort-dawn-patches-") as directory:
        root = Path(directory)
        for name in names:
            path = "cmake/" + name
            text = plan.read(path)
            touched = patch_paths(text)
            all_touched.update(touched)
            for relative in touched:
                target = root / relative
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read(relative).encode("utf-8"))
            before = {relative: (root / relative).read_text(encoding="utf-8") for relative in touched}
            patch_file = root / "input.patch"
            patch_file.write_bytes(text.encode("utf-8"))
            args = [
                executable,
                "--binary",
                "--batch",
                "--forward",
                "--ignore-whitespace",
                "--fuzz=0",
                "-p1",
                "--reject-file=-",
                "--no-backup-if-mismatch",
                "-i",
                str(patch_file),
            ]
            env = dict(os.environ, LC_ALL="C")
            check = subprocess.run(
                [*args, "--dry-run"], check=False, cwd=root, env=env, capture_output=True, text=True, timeout=30
            )
            fuzz = False
            if check.returncode:
                # Context drift is common in rolls. GNU patch can discard up to
                # two context lines, but never changes the patch's edited lines.
                args[args.index("--fuzz=0")] = "--fuzz=2"
                check = subprocess.run(
                    [*args, "--dry-run"], check=False, cwd=root, env=env, capture_output=True, text=True, timeout=30
                )
                fuzz = True
            if check.returncode:
                results.append({"path": path, "status": "conflict", "output": check.stdout + check.stderr})
                plan.report["blockers"].append(f"Patch conflict: {path}; fix or review whether it is already upstream")
                break
            applied = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=30, check=True)
            output = applied.stdout + applied.stderr
            refresh = fuzz or "offset" in output
            if refresh:
                expected = {relative: (root / relative).read_bytes() for relative in touched}
                refreshed = ""
                for relative in touched:
                    after = (root / relative).read_text(encoding="utf-8")
                    diff = "".join(
                        difflib.unified_diff(
                            before[relative].splitlines(keepends=True),
                            after.splitlines(keepends=True),
                            fromfile=f"a/{relative}",
                            tofile=f"b/{relative}",
                        )
                    )
                    if diff:
                        refreshed += f"diff --git a/{relative} b/{relative}\n" + diff
                if not refreshed:
                    raise ValueError(f"patch {path} produced no change; review removal")
                # Empty context lines need no leading space; avoid introducing
                # trailing whitespace into the checked-in patch files.
                refreshed = "".join("\n" if line == " \n" else line for line in refreshed.splitlines(keepends=True))
                # Verify the replacement patch reproduces the same source with
                # strict context before including it in the update.
                for relative, text_before in before.items():
                    (root / relative).write_bytes(text_before.encode("utf-8"))
                patch_file.write_bytes(refreshed.encode("utf-8"))
                strict_args = ["--fuzz=0" if arg.startswith("--fuzz=") else arg for arg in args]
                verified = subprocess.run(
                    strict_args, cwd=root, env=env, capture_output=True, text=True, timeout=30, check=False
                )
                diagnostics = verified.stdout + verified.stderr
                if (
                    verified.returncode
                    or re.search(r"\b(?:offset [+-]?\d+ lines?|with fuzz \d+)\b", diagnostics)
                    or any((root / relative).read_bytes() != data for relative, data in expected.items())
                ):
                    raise ValueError(f"refreshed patch {path} did not reproduce the expected source: {diagnostics}")
                plan.edit(path, refreshed)
            results.append(
                {"path": path, "status": "refreshed" if refresh else "clean", "contextFuzz": fuzz, "output": output}
            )
        plan.report["patchedFileSHA256"] = {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in sorted(all_touched)
        }
    plan.report["patches"] = results
