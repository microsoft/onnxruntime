# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Packages ONNX Runtime Java artifacts by combining native libraries from
various platform builds into final Java archive (JAR) files using 7z.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Add semver as a dependency
try:
    import semver
except ImportError:
    print("Error: The 'semver' package is not installed. Please add it to your requirements.txt.", file=sys.stderr)
    sys.exit(1)

# --- Helper Functions for Archiving ---


def find_7z_executable():
    """Finds the 7z executable, checking the system PATH and default installation locations."""
    # 1. Check if '7z' is in the PATH
    seven_zip_exe = shutil.which("7z")
    if seven_zip_exe:
        return seven_zip_exe

    # 2. Check if '7za' is in the PATH (common on Linux systems)
    seven_zip_exe = shutil.which("7za")
    if seven_zip_exe:
        return seven_zip_exe

    # 3. Check the default installation directory under Program Files
    program_files = os.environ.get("ProgramFiles")  # noqa: SIM112
    if program_files:
        default_path = Path(program_files) / "7-Zip" / "7z.exe"
        if default_path.is_file():
            return str(default_path)

    return None


SEVEN_ZIP_EXE = find_7z_executable()


def add_file_to_archive(archive_path: Path, file_to_add: Path, description: str):
    """Appends a single file to a zip archive (JAR file) using 7z."""
    print(f"  -> {description}...")
    try:
        if not SEVEN_ZIP_EXE:
            raise FileNotFoundError
        # Run 7z from the file's parent directory to ensure a clean archive path.
        subprocess.run(
            [SEVEN_ZIP_EXE, "a", str(archive_path), file_to_add.name],
            check=True,
            cwd=file_to_add.parent,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "Error: '7z' command not found. Please ensure 7-Zip is installed and in your PATH, or in the default location 'C:\\Program Files\\7-Zip'.",
            file=sys.stderr,
        )
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error: 7z failed to archive '{file_to_add.name}' to '{archive_path.name}'.", file=sys.stderr)
        print(f"Reason: {e.stderr}", file=sys.stderr)
        raise


def archive_directory_contents(archive_path: Path, source_dir: Path, description: str):
    """Archives a directory into a zip file (JAR file) using 7z, preserving its top-level name."""
    print(f"  -> {description}...")
    try:
        if not SEVEN_ZIP_EXE:
            raise FileNotFoundError
        # Run 7z from the parent of the source directory to ensure the source directory
        # itself is added to the archive, preserving the path structure (e.g., 'ai/...').
        subprocess.run(
            [SEVEN_ZIP_EXE, "a", str(archive_path), source_dir.name],
            check=True,
            cwd=source_dir.parent,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "Error: '7z' command not found. Please ensure 7-Zip is installed and in your PATH, or in the default location 'C:\\Program Files\\7-Zip'.",
            file=sys.stderr,
        )
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error: 7z failed to archive directory '{source_dir.name}' to '{archive_path.name}'.", file=sys.stderr)
        print(f"Reason: {e.stderr}", file=sys.stderr)
        raise


# --- Validation Helpers ---


def validate_version(version_string: str):
    """Validates if the version string conforms to the project's format."""
    print(f"Validating version string: {version_string}...")
    try:
        version_info = semver.Version.parse(version_string)
        if version_info.prerelease:
            prerelease_tag = version_info.prerelease
            allowed_tags_pattern = r"^(alpha|beta|rc)\d+$"
            if not re.match(allowed_tags_pattern, str(prerelease_tag)):
                raise ValueError(f"Pre-release tag '{prerelease_tag}' is not an allowed type.")
    except ValueError as e:
        print(f"Error: Version '{version_string}' is not valid. Reason: {e}", file=sys.stderr)
        print("Expected format is 'X.Y.Z' or 'X.Y.Z-(alpha|beta|rc)N'.", file=sys.stderr)
        sys.exit(1)
    print("Version format is valid.")


def validate_companion_jars(base_jar_path: Path):
    """Ensures that -sources.jar and -javadoc.jar files exist."""
    print("Validating presence of companion -sources.jar and -javadoc.jar...")
    base_stem = base_jar_path.stem
    directory = base_jar_path.parent
    sources_jar_path = directory / f"{base_stem}-sources.jar"

    if not sources_jar_path.is_file():
        print(f"Error: Missing companion sources JAR. Expected: {sources_jar_path.name}", file=sys.stderr)
        sys.exit(1)

    if not list(directory.glob(f"{base_stem}-javadoc*.jar")):
        print(f"Error: Missing companion javadoc JAR. Expected file like: {base_stem}-javadoc.jar", file=sys.stderr)
        sys.exit(1)
    print("Companion JARs are present.")


# --- Core Logic Function ---


def process_platform_archive(
    platform_path: Path,
    main_archive_file: Path,
    test_archive_file: Path,
    custom_lib_file: str,
    archive_custom_lib: bool,
):
    """Processes a single platform directory, adding only the 'ai' subdirectory to the main JAR."""
    print(f"Processing platform: {platform_path}...")

    # 1. Handle the custom op library.
    custom_lib_full_path = platform_path / custom_lib_file
    if custom_lib_file and custom_lib_full_path.is_file():
        if archive_custom_lib:
            add_file_to_archive(test_archive_file, custom_lib_full_path, f"Archiving '{custom_lib_file}' to test JAR")
        # Always remove the lib after processing to prevent it from being in the main JAR.
        print(f"  -> Removing '{custom_lib_file}' from source directory...")
        custom_lib_full_path.unlink()
    elif archive_custom_lib:
        # If we expected to archive the file but it wasn't there, it's a fatal error.
        print(f"Error: Expected custom op library '{custom_lib_file}' not found in {platform_path}", file=sys.stderr)
        sys.exit(1)

    # 2. Archive only the native library directory ('ai/...') to the main JAR.
    #    This explicitly excludes other files or folders like '_manifest'.
    native_lib_root = platform_path / "ai"
    if native_lib_root.is_dir():
        archive_directory_contents(
            main_archive_file, native_lib_root, f"Archiving native libs from '{native_lib_root.name}' to main JAR"
        )
    else:
        print(f"Warning: Native library path 'ai/' not found in {platform_path}. Skipping main archive step.")

    print(f"Finished platform: {platform_path}")
    print("--------------------------------")


def run_packaging(package_type: str, build_dir: str):
    """The main logic for the packaging process, refactored to be callable."""
    artifacts_base_dir = Path(build_dir) / "java-artifact"
    primary_package_dir = artifacts_base_dir / "onnxruntime-java-win-x64"
    if not primary_package_dir.is_dir():
        print(f"Error: Primary package directory not found at '{primary_package_dir}'", file=sys.stderr)
        sys.exit(1)

    # --- Version Discovery ---
    print(f"Discovering version from JAR files in '{primary_package_dir}'...")
    jar_pattern = str(primary_package_dir / "onnxruntime*-*.jar")
    jar_files = [Path(f) for f in glob.glob(jar_pattern) if "-sources" not in f and "-javadoc" not in f]
    if not jar_files:
        print(
            f"Error: Could not find a main JAR file in '{primary_package_dir}' to determine the version.",
            file=sys.stderr,
        )
        sys.exit(1)

    main_jar_file = jar_files[0]
    validate_companion_jars(main_jar_file)

    version = ""
    stem = main_jar_file.stem
    try:
        # Per user feedback, the version is everything after the first dash.
        _, version = stem.split("-", 1)
    except ValueError:
        # This will happen if there is no dash in the filename, which is unexpected.
        print(
            f"Error: Could not parse version from JAR file '{main_jar_file.name}'. Expected format <artifactId>-<version>.jar",
            file=sys.stderr,
        )
        sys.exit(1)

    if not version:
        print(
            f"Error: Could not parse version from JAR file '{main_jar_file.name}'. Version part is empty.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Version discovered: {version}")
    validate_version(version)

    # --- Package Definitions ---
    package_definitions: dict[str, dict[str, Any]] = {
        "cpu": {
            "platforms": [
                {"path": "onnxruntime-java-linux-x64", "lib": "libcustom_op_library.so", "archive_lib": True},
                {"path": "onnxruntime-java-linux-aarch64", "lib": "libcustom_op_library.so", "archive_lib": False},
                {"path": "onnxruntime-java-osx-arm64", "lib": "libcustom_op_library.dylib", "archive_lib": True},
            ]
        },
        "gpu": {
            "platforms": [{"path": "onnxruntime-java-linux-x64", "lib": "libcustom_op_library.so", "archive_lib": True}]
        },
    }

    # --- Processing Loop ---
    print(f"\n## Configuring for {package_type.upper()} package build...")

    final_main_archive = main_jar_file
    final_test_archive = primary_package_dir / "testing.jar"

    print(f"Using '{final_main_archive.name}' as the base for in-place packaging.")

    if not final_test_archive.is_file():
        print(f"Error: Base 'testing.jar' not found at '{final_test_archive}'.", file=sys.stderr)
        sys.exit(1)

    platforms_to_process = package_definitions[package_type]["platforms"]

    for platform in platforms_to_process:
        platform_full_path = artifacts_base_dir / platform["path"]
        if not platform_full_path.is_dir():
            print(f"Error: Required platform artifact directory not found: {platform_full_path}", file=sys.stderr)
            sys.exit(1)

        process_platform_archive(
            platform_path=platform_full_path,
            main_archive_file=final_main_archive,
            test_archive_file=final_test_archive,
            custom_lib_file=platform["lib"],
            archive_custom_lib=platform["archive_lib"],
        )

    print("\nScript completed successfully.")


def main():
    """Main script entry point for command-line execution."""
    if sys.platform != "win32":
        print("Error: This script is intended to be run on Windows.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Package ONNX Runtime Java artifacts.")
    parser.add_argument(
        "--package_type",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="The type of package to build ('cpu' or 'gpu').",
    )
    parser.add_argument(
        "--build_dir",
        type=str,
        help="The build directory containing the java-artifact folder.",
    )
    args = parser.parse_args()

    build_dir = args.build_dir
    if not build_dir:
        try:
            build_dir = os.environ["BUILD_BINARIESDIRECTORY"]
        except KeyError:
            print(
                "Error: Environment variable BUILD_BINARIESDIRECTORY is not set and --build_dir is not provided.",
                file=sys.stderr,
            )
            sys.exit(1)

    run_packaging(args.package_type, build_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}", file=sys.stderr)
        sys.exit(1)
