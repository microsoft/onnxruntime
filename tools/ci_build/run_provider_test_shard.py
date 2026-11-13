# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run one of four provider-test shards with isolated writable state."""

import argparse
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def run_shard(binary, shard_index, report, test_args):
    binary = binary.resolve(strict=True)
    report = report.resolve()
    report.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"provider-shard-{shard_index}-", dir=binary.parent) as directory:
        root = Path(directory)
        work = root / "work"
        work.mkdir()
        temporary = root / "tmp"
        temporary.mkdir()
        # Keep ../models and testdata lookups equivalent to execution beside the binary.
        (work / "testdata").symlink_to(binary.parent / "testdata", target_is_directory=True)
        (root / "models").symlink_to(binary.parent.parent / "models", target_is_directory=True)
        for library in binary.parent.glob("*.so*"):
            (work / library.name).symlink_to(library)
        status = root / "shard-status"
        env = os.environ.copy()
        env.update(
            GTEST_TOTAL_SHARDS="4",
            GTEST_SHARD_INDEX=str(shard_index),
            GTEST_SHARD_STATUS_FILE=str(status),
            TMPDIR=str(temporary),
            TMP=str(temporary),
            TEMP=str(temporary),
        )
        result = subprocess.run(
            [str(binary), *test_args, f"--gtest_output=xml:{report}"],
            cwd=work,
            env=env,
            check=False,
        )
        if result.returncode:
            return result.returncode
        if not status.is_file():
            raise RuntimeError("The provider test binary did not acknowledge GoogleTest sharding")
        executed = [case for case in ET.parse(report).iter("testcase") if case.get("status") == "run"]
        if not executed:
            raise RuntimeError(f"Provider test shard {shard_index} executed zero tests")
        print(f"Provider shard {shard_index}: {len(executed)} executed tests; report: {report}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int, choices=range(4))
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("test_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    test_args = args.test_args[1:] if args.test_args[:1] == ["--"] else args.test_args
    return run_shard(args.binary, args.shard_index, args.report, test_args)


if __name__ == "__main__":
    raise SystemExit(main())
