# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Inventory built artifacts only; never infer validation from a filename or target."""

import argparse
import hashlib
import json
from pathlib import Path


def inventory(artifacts, version, platform, variant):
    entries = []
    seen = set()
    for artifact in artifacts:
        label, separator, name = artifact.partition("=")
        if not separator or not label or not name:
            raise ValueError(f"Invalid artifact {artifact!r}; pass TARGET=FILE from a built release target.")
        path = Path(name)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        artifact_entries = 0
        for file in files:
            if file.is_dir():
                continue
            if not file.is_file():
                raise ValueError(f"Release artifact {file} is not a regular file; package its payload explicitly.")
            relative = file.relative_to(path.parent).as_posix()
            identity = (label, relative)
            if identity in seen:
                raise ValueError(f"Duplicate release artifact {identity}; use distinct output names per target.")
            seen.add(identity)
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            entries.append(
                {
                    "target": label,
                    "file": relative,
                    "bytes": file.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
            artifact_entries += 1
        if not artifact_entries:
            raise ValueError(f"Release artifact {path} is empty; build a populated package before inventorying it.")
    if not entries:
        raise ValueError("Release inventory is empty; provide at least one built language package.")
    return {
        "schema_version": 1,
        "version": version,
        "platform": platform,
        "variant": variant,
        "validation": "artifact-built-only; consumer/native validation is recorded separately",
        "artifacts": sorted(entries, key=lambda entry: (entry["target"], entry["file"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--version-file", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(
        args.artifact, args.version_file.read_text(encoding="utf-8").strip(), args.platform, args.variant
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
