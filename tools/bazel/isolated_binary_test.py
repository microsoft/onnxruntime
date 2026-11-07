# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run existing loader probes without modifying executable/build runfiles."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--library", action="append", type=Path, default=[])
    parser.add_argument("--fixture", action="append", type=Path, default=[])
    parser.add_argument("--testdata", type=Path)
    args = parser.parse_args()
    temporary = os.environ.get("TEST_TMPDIR")
    if not temporary:
        parser.error("TEST_TMPDIR is missing; run this probe with bazel test.")

    directory = Path(temporary).resolve() / "runtime"
    try:
        directory.mkdir()
        executable = directory / args.binary.name
        shutil.copy2(args.binary, executable)
        for source in args.library:
            shutil.copy2(source, directory / source.name)
        if args.fixture:
            fixtures = directory / "fixtures"
            fixtures.mkdir()
            for source in args.fixture:
                shutil.copy2(source, fixtures / source.name)
        if args.testdata:
            testdata = directory / "testdata"
            testdata.mkdir()
            shutil.copy2(args.testdata, testdata / args.testdata.name)
        environment = os.environ.copy()
        environment["TEST_TMPDIR"] = str(directory)
        environment["ORT_RUNNING_UNIT_TESTS"] = "1"
        return subprocess.run([str(executable)], cwd=directory, env=environment, check=False).returncode
    except OSError as error:
        parser.error(
            f"Cannot prepare or execute loader probe {args.binary}: {error}. "
            "Check declared binary/library runfiles and writable TEST_TMPDIR space."
        )


if __name__ == "__main__":
    sys.exit(main())
