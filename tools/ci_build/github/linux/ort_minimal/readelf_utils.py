#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Utilities to help analyze the sections in a binary using readelf.
"""

import argparse
import collections
import os
import re
import subprocess
import sys


def get_section_sizes(binary_path, readelf_path, dump_to_file=None):
    """
    Get the size of each section using readelf.
    :param binary_path: Path to binary to analyze.
    :param readelf_path: Path to readelf binary. Default is 'readelf'.
    :param dump_to_file: File object to write section sizes and diagnostic info to. Defaults to None.
    :return:
    """

    cmd = [readelf_path, "--sections", "--wide", binary_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)  # noqa: PLW1510
    result.check_returncode()
    output = result.stdout.decode("utf-8")

    section_sizes = {}

    # Parse output in this format:
    #   [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
    for match in re.finditer(r"\[[\s\d]+\] (\..*)$", output, re.MULTILINE):
        items = match.group(1).split()
        name = items[0]
        # convert size from hex to int
        size = int(items[4], 16)
        section_sizes[name] = size

        if dump_to_file:
            print(f"{name}:{size}", file=dump_to_file)

    return section_sizes


def diff_sections_total_size(base_binary_path, binary_path, readelf_path="readelf"):
    """
    Diff the sections entries for two binaries.
    :param base_binary_path: Path to base binary for diff.
    :param binary_path: Path to binary to diff using.
    :param readelf_path: Path to 'readelf' binary. Defaults to 'readelf'
    :return: Ordered dictionary containing size of diff for all sections with a diff, the diff for the sum of the
             sections in the 'Sections total' entry, and the diff for the on-disk file size in the 'File size' entry
    """

    filesize = os.path.getsize(binary_path)
    base_filesize = os.path.getsize(base_binary_path)

    section_sizes = get_section_sizes(binary_path, readelf_path)
    base_section_sizes = get_section_sizes(base_binary_path, readelf_path)
    merged_keys = set(base_section_sizes.keys()) | set(section_sizes.keys())

    base_total = 0
    total = 0
    results = collections.OrderedDict()

    for section in sorted(merged_keys):
        base_size = base_section_sizes.get(section, 0)
        size = section_sizes.get(section, 0)

        base_total += base_size
        total += size

        if size != base_size:
            results[section] = size - base_size

    results["Sections total"] = total - base_total
    results["File size"] = filesize - base_filesize

    return results


def main():
    argparser = argparse.ArgumentParser(
        description="Analyze sections in a binary using readelf. "
        "Perform a diff between two binaries if --base_binary_path is specified."
    )

    argparser.add_argument("-r", "--readelf_path", type=str, help="Path to readelf executable.")
    argparser.add_argument(
        "-b",
        "--base_binary_path",
        type=os.path.realpath,
        default=None,
        help="Path to base binary if performing a diff between two binaries.",
    )
    argparser.add_argument(
        "-w", "--write_to", type=str, default=None, help="Path to write output to. Writes to stdout if not provided."
    )
    argparser.add_argument("binary_path", type=os.path.realpath, help="Shared library to analyze.")

    args = argparser.parse_args()

    out_file = sys.stdout
    if args.write_to:
        out_file = open(args.write_to, "w")  # noqa: SIM115

    if args.base_binary_path:
        diffs = diff_sections_total_size(args.base_binary_path, args.binary_path, args.readelf_path)
        for key, value in diffs.items():
            print(f"{key}:{value}", file=out_file)
    else:
        section_sizes = get_section_sizes(args.binary_path, args.readelf_path, out_file)
        filesize = os.path.getsize(args.binary_path)
        print(f"Sections total:{sum(section_sizes.values())}", file=out_file)
        print(f"File size:{filesize}", file=out_file)

    if args.write_to:
        out_file.close()


if __name__ == "__main__":
    main()
