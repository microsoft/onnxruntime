#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#We have some Azure DevOps pipelines that each has two parts: the first part build source code into binary then upload it, then second part download the binary and run tests. Because the second part may need GPUs. We use CMake and CTest to run tests. However, in the first part cmake generates a file called CTestTestfile.cmake. And it has hardcoded file paths. This script is to update the abs paths.

import argparse
import os
import re
import sys
from pathlib import Path

def update_ctest_paths(file_path: Path, new_repo_dir: Path, new_build_dir: Path):
    """
    Finds and replaces hardcoded source and build directory paths
    in a CTestTestfile.cmake file.

    Args:
        file_path: The path to the CTestTestfile.cmake file.
        new_repo_dir: The path to the new repository root directory.
        new_build_dir: The path to the new build directory.
    """
    print(f"Processing file: {file_path}")

    # 1. Define regex to find the old paths from the header comments
    src_dir_pattern = re.compile(r"# Source directory: (.*)")
    build_dir_pattern = re.compile(r"# Build directory: (.*)")

    old_src_dir = None
    old_build_dir = None

    # 2. Read the file to extract old paths
    with file_path.open('r', encoding='utf-8') as f:
        # The paths are usually in the first few lines
        for line in f:
            if not old_src_dir and (src_match := src_dir_pattern.match(line)):
                # The path in the file points to a subdirectory (e.g., .../s/cmake)
                # The actual source root is its parent.
                cmake_subdir = Path(src_match.group(1).strip())
                old_src_dir = str(cmake_subdir.parent)
                print(f"Found CMake source sub-directory: '{cmake_subdir}'")
            elif not old_build_dir and (build_match := build_dir_pattern.match(line)):
                old_build_dir = build_match.group(1).strip()

            if old_src_dir and old_build_dir:
                break

    if not old_src_dir or not old_build_dir:
        print(f"Error: Could not find old source/build directory paths in {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Deduced old source root: '{old_src_dir}'")
    print(f"Found old build directory: '{old_build_dir}'")

    # 3. Read the entire file content for replacement
    content = file_path.read_text(encoding='utf-8')

    # 4. Normalize new paths to use forward slashes (CMake's preference)
    new_src_posix = new_repo_dir.as_posix()
    new_build_posix = new_build_dir.as_posix()

    print(f"Replacing with new source directory: '{new_src_posix}'")
    print(f"Replacing with new build directory: '{new_build_posix}'")

    # 5. Perform replacements for both path styles (forward and escaped backslashes)
    # It's safer to replace the longer, more specific build path first.
    if old_build_dir in content:
        content = content.replace(old_build_dir, new_build_posix)
        # Handle the escaped backslash variant (e.g., "C:\\path\\to\\build")
        content = content.replace(old_build_dir.replace('/', '\\\\'), new_build_posix.replace('/', '\\\\'))

    if old_src_dir in content:
        content = content.replace(old_src_dir, new_src_posix)
        # Handle the escaped backslash variant
        content = content.replace(old_src_dir.replace('/', '\\\\'), new_src_posix.replace('/', '\\\\'))

    # 6. Write the updated content back to the file
    file_path.write_text(content, encoding='utf-8')
    print(f"Successfully updated paths in {file_path}")


def main():
    """Parses command-line arguments and runs the update logic."""
    # Determine the repository root based on the script's location
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

    parser = argparse.ArgumentParser(
        description="Update hardcoded paths in a CTestTestfile.cmake file for relocatable test execution.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "ctest_file",
        type=Path,
        help="Path to the CTestTestfile.cmake file to be updated."
    )
    parser.add_argument(
        "new_build_dir",
        type=Path,
        help="The new build directory path (e.g., $(Pipeline.Workspace)/b)."
    )

    args = parser.parse_args()

    if not args.ctest_file.is_file():
        print(f"Error: CTest file not found at '{args.ctest_file}'", file=sys.stderr)
        sys.exit(1)

    try:
        update_ctest_paths(args.ctest_file, Path(REPO_DIR), args.new_build_dir)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()