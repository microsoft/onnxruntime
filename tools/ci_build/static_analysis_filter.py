# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import collections
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filters MSBuild output for static analysis warnings.")
    parser.add_argument("input",
                        help="Input file containing the build output.")
    parser.add_argument("--tsv",
                        help="Writes TSV output to the specified file.")
    parser.add_argument("--echo", action="store_true",
                        help="Echoes input to stdout.")
    return parser.parse_args()

WarningInfo = collections.namedtuple(
    "WarningInfo", ["location", "code", "description", "project"])

def make_warning_info(location, code_str, description, project):
    normalized_location = location.strip().lower()
    code = int(code_str)
    return WarningInfo._make([
        normalized_location, code, description, project])

path_pattern = r"[A-Za-z0-9_\- :./\\]+"
warning_line_pattern = r"^(" + path_pattern + r"\(\d+\)):" + \
                       r"\s+warning C(\d+):" + \
                       r"\s+(.*)" + \
                       r"\s+\[(" + path_pattern + r")\]$"
warning_line_re = re.compile(warning_line_pattern)

def parse_warning_line(line):
    match = warning_line_re.match(line)
    if not match: return None
    return make_warning_info(match.group(1), match.group(2),
                             match.group(3), match.group(4))

def filter_warning(warning, min_warning_code, ignored_location_pattern_res):
    if warning.code < min_warning_code: return False
    for ignored_location_pattern_re in ignored_location_pattern_res:
        if ignored_location_pattern_re.search(warning.location):
            return False
    return True

def write_tsv(warnings, tsv_file):
    with open(tsv_file, "w") as tsv_file_content:
        tsv_file_content.write("location\tcode\tdescription\tproject\n")
        for warning in warnings:
            tsv_file_content.write(
                "\t".join([warning.location,
                           str(warning.code),
                           warning.description,
                           warning.project]) +
                "\n")

def main():
    args = parse_args()
    path_separator_pattern = r"[\\/]"
    ignored_location_pattern_res = [re.compile(pattern) for pattern in [
        path_separator_pattern.join(["protobuf", "src"]),
    ]]
    warnings = list()
    unique_warnings = set()

    with open(args.input, "r") as infile:
        for line in infile:
            warning = parse_warning_line(line)
            if warning and \
               filter_warning(warning, 6000, ignored_location_pattern_res) and \
               warning not in unique_warnings:
                warnings.append(warning)
                unique_warnings.add(warning)

            if args.echo:
                sys.stdout.write(line)

    if args.tsv:
        write_tsv(warnings, args.tsv)

    return 0

if __name__ == "__main__":
    sys.exit(main())
