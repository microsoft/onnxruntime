# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Compare shipped native payloads with the declared toolchain strip outputs."""

import argparse
import hashlib
import io
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def _members(name, contents):
    if name.endswith((".zip", ".jar", ".whl", ".nupkg", ".snupkg", ".aar")):
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            for entry in archive.infolist():
                if not entry.is_dir():
                    yield from _members(f"{name}!/{entry.filename}", archive.read(entry))
    elif name.endswith((".tgz", ".tar.gz")):
        with tarfile.open(fileobj=io.BytesIO(contents), mode="r:gz") as archive:
            for entry in archive:
                if entry.isfile():
                    source = archive.extractfile(entry)
                    if source is None:
                        raise AssertionError(
                            f"Cannot read regular archive member {name}!/{entry.name}; rebuild the package."
                        )
                    with source:
                        yield from _members(f"{name}!/{entry.name}", source.read())
    else:
        yield name, contents


def _package_members(path):
    sources = sorted(path.rglob("*")) if path.is_dir() else [path]
    for source in sources:
        if source.is_file():
            yield from _members(str(source), source.read_bytes())


def verify(package, binaries):
    expected = {path.name: hashlib.sha256(path.read_bytes()).digest() for path in binaries}
    if len(expected) != len(binaries):
        raise AssertionError(f"Native release inputs for {package} must have distinct basenames: {binaries}.")
    seen = set()
    count = 0
    for name, contents in _package_members(package):
        basename = PurePosixPath(name.replace("\\", "/")).name
        if basename in expected:
            digest = hashlib.sha256(contents)
            if digest.digest() != expected[basename]:
                raise AssertionError(
                    f"{name} differs from its declared release binary. Wire the package to the "
                    "toolchain strip output, not the raw link artifact or a manually stripped copy."
                )
            seen.add(basename)
            count += 1
            print(f"{name}: {len(contents)} bytes, SHA256 {digest.hexdigest()}")
        elif basename.endswith((".so", ".dylib", ".dll", ".node", ".pyd")) or ".so." in basename:
            if basename != "Microsoft.ML.OnnxRuntime.dll":
                raise AssertionError(
                    f"{name} is an undeclared native payload; add its release action output to the test."
                )
    if seen != set(expected):
        raise AssertionError(f"{package} is missing declared release payloads: {sorted(set(expected) - seen)}.")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", nargs="+", required=True, metavar="PATH")
    parser.add_argument("--cuda-binary", type=Path)
    args = parser.parse_args()
    count = 0
    for group in args.package:
        if len(group) < 2:
            parser.error("--package requires the package output followed by its declared release binaries.")
        package, *binaries = map(Path, group)
        if args.cuda_binary:
            binaries.append(args.cuda_binary)
        count += verify(package, binaries)
    print(f"Verified {count} native payloads across {len(args.package)} package outputs against release action bytes.")


if __name__ == "__main__":
    main()
