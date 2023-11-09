# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import os
import pathlib
import shutil
import tempfile

from util import logger, run

_log = logger.get_logger("fix_long_lines", logging.INFO)


# look for long lines in the file, and if found run clang-format on those lines
def _process_files(filenames, clang_exe, tmpdir):
    for path in filenames:
        _log.debug(f"Checking {path}")
        bad_lines = []

        with open(path, encoding="UTF8") as f:
            line_num = 0
            for line in f:
                line_num += 1  # clang-format line numbers start at 1
                if len(line) > 120:
                    bad_lines.append(line_num)

        if bad_lines:
            _log.info(f"Updating {path}")
            filename = os.path.basename(path)
            target = os.path.join(tmpdir, filename)
            shutil.copy(path, target)

            # run clang-format to update just the long lines in the file
            cmd = [
                clang_exe,
                "-i",
            ]
            for line in bad_lines:
                cmd.append(f"--lines={line}:{line}")

            cmd.append(target)

            run(*cmd, cwd=tmpdir, check=True, shell=True)

            # copy updated file back to original location
            shutil.copy(target, path)


# file extensions we process
_EXTENSIONS = [".cc", ".h"]


def _get_branch_diffs(ort_root, branch):
    command = ["git", "diff", branch, "--name-only"]
    result = run(*command, capture_stdout=True, check=True)

    # stdout is bytes. one filename per line. decode, split, and filter to the extensions we are looking at
    for f in result.stdout.decode("utf-8").splitlines():
        if os.path.splitext(f.lower())[-1] in _EXTENSIONS:
            yield os.path.join(ort_root, f)


def _get_file_list(path):
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file.lower())[-1] in _EXTENSIONS:
                yield os.path.join(root, file)


def main():
    argparser = argparse.ArgumentParser(
        "Script to fix long lines in the source using clang-format. "
        "Only lines that exceed the 120 character maximum are altered in order to minimize the impact. "
        "Checks .cc and .h files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    argparser.add_argument(
        "--branch",
        type=str,
        default="origin/main",
        help="Limit changes to files that differ from this branch. Use origin/main when preparing a PR.",
    )

    argparser.add_argument(
        "--all_files",
        action="store_true",
        help="Process all files under /include/onnxruntime and /onnxruntime/core. Ignores --branch value.",
    )

    argparser.add_argument(
        "--clang-format",
        type=pathlib.Path,
        required=False,
        default="clang-format",
        help="Path to clang-format executable",
    )

    argparser.add_argument("--debug", action="store_true", help="Set log level to DEBUG.")

    args = argparser.parse_args()

    if args.debug:
        _log.setLevel(logging.DEBUG)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    ort_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    with tempfile.TemporaryDirectory() as tmpdir:
        # create config in tmpdir
        with open(os.path.join(tmpdir, ".clang-format"), "w") as f:
            f.write(
                """
            BasedOnStyle: Google
            ColumnLimit: 120
            DerivePointerAlignment: false
            """
            )

        clang_format = str(args.clang_format)

        if args.all_files:
            include_path = os.path.join(ort_root, "include", "onnxruntime")
            src_path = os.path.join(ort_root, "onnxruntime", "core")
            _process_files(_get_file_list(include_path), clang_format, tmpdir)
            _process_files(_get_file_list(src_path), clang_format, tmpdir)
        else:
            _process_files(_get_branch_diffs(ort_root, args.branch), clang_format, tmpdir)


if __name__ == "__main__":
    main()
