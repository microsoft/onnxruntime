#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file replaces https URLs in deps.txt to local file paths. It runs after we download the dependencies from Azure
# DevOps Artifacts

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dep:
    name: str
    url: str
    sha1_hash: str


def parse_arguments():
    parser = argparse.ArgumentParser()
    # The directory that contains downloaded zip files
    parser.add_argument("--new_dir", required=False)

    return parser.parse_args()


def main():
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

    args = parse_arguments()
    new_dir = None
    if args.new_dir:
        new_dir = Path(args.new_dir)
    else:
        BUILD_BINARIESDIRECTORY = os.environ.get("BUILD_BINARIESDIRECTORY")
        if BUILD_BINARIESDIRECTORY is None:
            raise NameError("Please specify --new_dir or set the env var BUILD_BINARIESDIRECTORY")
        new_dir = Path(BUILD_BINARIESDIRECTORY) / "deps"

    if not new_dir.exists():
        raise NameError("The local dir %s does not exist" % (str(new_dir)))

    deps = []

    csv_file_path = Path(REPO_DIR) / "cmake" / "deps.txt"

    # Read the whole file into memory first
    with csv_file_path.open("r", encoding="utf-8") as f:
        depfile_reader = csv.reader(f, delimiter=";")
        for row in depfile_reader:
            if len(row) != 3:
                continue
            # Lines start with "#" are comments
            if row[0].startswith("#"):
                continue
            deps.append(Dep(row[0], row[1], row[2]))

    # Write updated content back
    with csv_file_path.open("w", newline="", encoding="utf-8") as f:
        depfile_writer = csv.writer(f, delimiter=";")
        for dep in deps:
            if dep.url.startswith("https://"):
                new_url = new_dir / dep.url[8:]
                if not new_url.exists():
                    # Write the original thing back
                    depfile_writer.writerow([dep.name, dep.url, dep.sha1_hash])
                    continue
                depfile_writer.writerow([dep.name, new_url.as_posix(), dep.sha1_hash])
            else:
                # Write the original thing back
                depfile_writer.writerow([dep.name, dep.url, dep.sha1_hash])


if __name__ == "__main__":
    main()
