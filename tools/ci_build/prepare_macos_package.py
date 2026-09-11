# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pathlib
import shutil
import stat as stat_module
import subprocess
import sys
import tarfile
from datetime import datetime


def run_command(command: list[str | pathlib.Path], check: bool = True) -> subprocess.CompletedProcess:
    """Helper to run a command, stream its output, and check for errors."""
    print(f"Executing: {' '.join(map(str, command))}", flush=True)
    try:
        return subprocess.run(command, check=check, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}", file=sys.stderr)
        print(f"--- STDOUT ---\n{e.stdout}", file=sys.stderr)
        print(f"--- STDERR ---\n{e.stderr}", file=sys.stderr)
        raise


def get_relative_file_paths(root_dir: pathlib.Path) -> set[pathlib.Path]:
    """
    Returns a set of all relative file paths within a directory,
    ignoring any files inside .dSYM directories.
    """
    paths = set()
    for p in root_dir.rglob("*"):
        # Check if any part of the path is a .dSYM directory.
        if any(part.endswith(".dSYM") for part in p.relative_to(root_dir).parts):
            continue
        if p.is_file():
            paths.add(p.relative_to(root_dir))
    return paths


def is_macho_binary(file_path: pathlib.Path) -> bool:
    """Checks if a file is a Mach-O binary using the 'file' command."""
    if not file_path.is_file():
        return False
    try:
        result = run_command(["file", file_path])
        return "Mach-O" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    """Main function to prepare macOS packages for signing."""
    # 1. Setup paths and parse arguments
    parser = argparse.ArgumentParser(description="Prepares macOS packages for signing.")
    parser.add_argument(
        "--staging_dir",
        type=pathlib.Path,
        required=True,
        help="The directory where artifacts are staged and processed.",
    )
    args = parser.parse_args()
    staging_dir = args.staging_dir.resolve()

    if not staging_dir.is_dir():
        raise FileNotFoundError(f"Staging directory not found: {staging_dir}")

    os.chdir(staging_dir)
    print(f"##[group]Working in directory: {staging_dir}")
    print(f"Initial contents: {[p.name for p in staging_dir.iterdir()]}")
    print("##[endgroup]")

    # 2. Unpack all .tgz archives
    print("##[group]Unpacking downloaded archives...")
    tgz_files = list(staging_dir.glob("*.tgz"))
    if not tgz_files:
        raise FileNotFoundError("Build Error: No .tgz files found to process.")

    for tgz in tgz_files:
        print(f"Extracting {tgz.name}...")
        with tarfile.open(tgz) as tar:
            tar.extractall(path=".")
        tgz.unlink()  # Delete the archive
    print("##[endgroup]")

    # 3. Locate architecture-specific directories
    print("##[group]Locating architecture directories...")
    arm64_dirs = list(staging_dir.glob("onnxruntime-osx-arm64*"))
    x64_dirs = list(staging_dir.glob("onnxruntime-osx-x86_64*"))

    if len(arm64_dirs) != 1 or len(x64_dirs) != 1:
        raise FileNotFoundError(
            f"Build Error: Expected 1 arm64 and 1 x64 directory, but found: arm64={len(arm64_dirs)}, x64={len(x64_dirs)}"
        )

    arm64_dir, x64_dir = arm64_dirs[0], x64_dirs[0]
    print(f"Found ARM64 source: {arm64_dir.name}")
    print(f"Found x86_64 source: {x64_dir.name}")
    print("##[endgroup]")

    # **NEW**: Remove _manifest directories before comparison or processing.
    print("##[group]Removing _manifest directories...")
    for package_dir in (arm64_dir, x64_dir):
        manifest_path = package_dir / "_manifest"
        if manifest_path.is_dir():
            print(f"Removing manifest directory: {manifest_path.relative_to(staging_dir)}")
            shutil.rmtree(manifest_path)
    print("##[endgroup]")

    # 4. Error Check: Verify file tree structures are identical
    print("##[group]Verifying file tree structures...")
    arm64_files = get_relative_file_paths(arm64_dir)
    x64_files = get_relative_file_paths(x64_dir)

    if arm64_files != x64_files:
        difference = arm64_files.symmetric_difference(x64_files)
        print(f"ERROR: File tree structures do not match. Found {len(difference)} differing files:", file=sys.stderr)
        for f in sorted(difference):
            print(f"- {f}", file=sys.stderr)
        sys.exit(1)

    print("✅ File tree structures match.")
    print("##[endgroup]")

    # 5. Create the universal binary package
    print("##[group]Creating universal2 package with lipo...")
    universal_dir = staging_dir / arm64_dir.name.replace("arm64", "universal2")

    print(f"Copying {arm64_dir.name} to {universal_dir.name} as a template.")
    shutil.copytree(arm64_dir, universal_dir, symlinks=True, ignore=shutil.ignore_patterns("*.dSYM"))

    for relative_path in arm64_files:
        arm64_file = arm64_dir / relative_path
        x64_file = x64_dir / relative_path
        universal_file = universal_dir / relative_path

        if is_macho_binary(arm64_file) and is_macho_binary(x64_file):
            print(f"Combining {relative_path}...")
            run_command(["lipo", "-create", arm64_file, x64_file, "-output", universal_file])
            run_command(["lipo", "-info", universal_file])
    print("##[endgroup]")

    # Remove .dSYM folders from source packages before zipping.
    print("##[group]Removing .dSYM folders from source packages...")
    for package_dir in (arm64_dir, x64_dir):
        for dsym_dir in package_dir.rglob("*.dSYM"):
            if dsym_dir.is_dir():
                print(f"Removing {dsym_dir.relative_to(staging_dir)}")
                shutil.rmtree(dsym_dir)
    print("##[endgroup]")

    # 6. Zip all packages for signing and clean up
    print("##[group]Zipping all packages for signing...")
    for dir_path in (arm64_dir, x64_dir, universal_dir):
        # Create a zip file in the staging directory.
        zip_file_path = staging_dir / f"{dir_path.name}.zip"
        print(f"Zipping {dir_path.name} to {zip_file_path}")
        # The source directory path (dir_path.name) is relative to the current working directory (staging_dir).
        run_command(["zip", "-FSr", "--symlinks", zip_file_path, dir_path.name])

        print(f"Removing directory {dir_path.name}")
        shutil.rmtree(dir_path)

    print("Final contents of staging directory:")
    for item in sorted(staging_dir.iterdir()):
        try:
            stat = item.stat()
            size = stat.st_size
            mode_str = stat_module.filemode(stat.st_mode)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%b %d %H:%M")
            print(f"{mode_str} {size:>10} {mtime} {item.name}")
        except FileNotFoundError:
            # Handle cases where a file might be a broken symlink
            print(f"l????????? {'?':>10} ? ? {item.name} (broken link)")

    print("##[endgroup]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"##[error]A critical error occurred: {e}", file=sys.stderr)
        sys.exit(1)
