# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create deterministic files for a Go module proxy."""

import argparse
import json
import re
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--version-file", type=Path, required=True)
    parser.add_argument("--source", type=Path, nargs="*", default=[])
    parser.add_argument("--legal", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    module_text = args.module.read_text(encoding="utf-8")
    match = re.search(r"(?m)^module\s+(\S+)\s*$", module_text)
    if not match:
        raise ValueError("go.mod does not define a module path.")
    module_path = match.group(1)
    version = "v" + args.version_file.read_text(encoding="utf-8").strip()
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise ValueError(f"Go proxy output is not a directory: {args.output_dir}")
        if any(args.output_dir.iterdir()):
            raise ValueError(f"Go proxy output directory is not empty: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)

    output_dir = args.output_dir / module_path / "@v"
    output_dir.mkdir(parents=True, exist_ok=True)
    mod_output = output_dir / f"{version}.mod"
    info_output = output_dir / f"{version}.info"
    zip_output = output_dir / f"{version}.zip"

    mod_output.write_text(module_text, encoding="utf-8", newline="\n")
    info_output.write_text(
        json.dumps({"Version": version, "Time": "1970-01-01T00:00:00Z"}, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # Keep logical action paths: resolving follows Bazel input symlinks out of the
    # execroot and makes otherwise related module/source paths incomparable.
    module_dir = args.module.absolute().parent
    entries = {"go.mod": module_text.encode()}
    for source in args.source:
        logical_source = source.absolute()
        entries[logical_source.relative_to(module_dir).as_posix()] = source.read_bytes()
    for legal in args.legal:
        entries[legal.name] = legal.read_bytes()

    root = f"{module_path}@{version}/"
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, data in sorted(entries.items()):
            info = zipfile.ZipInfo(root + relative, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)


if __name__ == "__main__":
    main()
