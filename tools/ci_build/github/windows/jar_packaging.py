# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Packages ONNX Runtime Java artifacts by combining native libraries from
various platform builds into final Java archive (JAR) files.

This script is designed for CI/CD pipelines and expects a specific input
directory structure created by preceding build steps.

Input Format:
The script requires a root directory named 'java-artifact/'. This directory
must contain a specific structure of subdirectories and files:

1.  **Platform-Specific Subdirectories:** For each target platform, a
    subdirectory must exist. Examples:
    - `java-artifact/onnxruntime-java-win-x64/`
    - `java-artifact/onnxruntime-java-linux-x64/`
    - `java-artifact/onnxruntime-java-osx-arm64/`

2.  **Native Library Path:** Inside each platform subdirectory, the native
    binaries (.dll, .so, .dylib) must be placed in a nested path that
    matches the Java package structure.
    - Path format: `ai/onnxruntime/native/<platform-arch>/`
    - Example: The `onnxruntime.dll` for Windows x64 must be at:
      `java-artifact/onnxruntime-java-win-x64/ai/onnxruntime/native/win-x64/onnxruntime.dll`

3.  **Version Source JAR:** The primary Windows directory
    (`onnxruntime-java-win-x64`) must also contain the initial Java JAR
    (e.g., 'onnxruntime-1.18.0.jar'). The script reads this file's name
    to determine the package version. This JAR should contain the compiled
    Java class files.

Outputs:
- A primary ONNX Runtime JAR file (e.g., 'onnxruntime-1.18.0.jar' or
  'onnxruntime_gpu-1.18.0.jar'). This is a comprehensive archive containing
  the native libraries for all supported platforms.
- A secondary testing JAR file (e.g., 'testing-onnxruntime.jar') that
  includes custom operator libraries used for validation.
"""

import argparse
import os
import glob
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Any

# --- Helper Functions for Archiving ---

def add_file_to_archive(archive_path: Path, file_to_add: Path, description: str):
    """Appends a single file to a zip archive (JAR file)."""
    print(f"  -> {description}...")
    try:
        # Open in append mode 'a' to add files to an existing archive.
        with zipfile.ZipFile(archive_path, 'a', compression=zipfile.ZIP_DEFLATED) as zf:
            # arcname ensures the path inside the zip is just the filename,
            # not the full path from the filesystem.
            zf.write(file_to_add, arcname=file_to_add.name)
    except Exception as e:
        print(f"Error: Failed to archive '{file_to_add.name}' to '{archive_path.name}'.", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        raise

def archive_directory_contents(archive_path: Path, source_dir: Path, description: str):
    """Archives all contents of a directory into a zip file (JAR file)."""
    print(f"  -> {description}...")
    try:
        # Open in append mode 'a' to add files from multiple platforms
        # into the same archive.
        with zipfile.ZipFile(archive_path, 'a', compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    # Create a relative path for the arcname to preserve the
                    # directory structure within the zip file.
                    arcname = file_path.relative_to(source_dir)
                    zf.write(file_path, arcname=arcname)
    except Exception as e:
        print(f"Error: Failed to archive contents of '{source_dir}' to '{archive_path.name}'.", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        raise


# --- Core Logic Function ---

def process_platform_archive(
    platform_path: Path,
    main_archive_file: Path,
    test_archive_file: Path,
    custom_lib_file: str,
    archive_custom_lib: bool
):
    """
    Processes a single platform directory, handling all archiving logic.

    Args:
        platform_path: Path to the platform-specific directory.
        main_archive_file: Path to the final main JAR file.
        test_archive_file: Path to the final test JAR file.
        custom_lib_file: Filename of the custom operator library.
        archive_custom_lib: Boolean flag to archive the custom library.
    """
    print(f"Processing platform: {platform_path}...")

    if not platform_path.is_dir():
        print(f"Error: Directory not found: {platform_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Handle the custom library file if it exists.
    custom_lib_path = platform_path / custom_lib_file
    if custom_lib_file and custom_lib_path.is_file():
        if archive_custom_lib:
            add_file_to_archive(
                archive_path=test_archive_file,
                file_to_add=custom_lib_path,
                description=f"Archiving '{custom_lib_file}' to test JAR"
            )

        print(f"  -> Removing '{custom_lib_file}'...")
        custom_lib_path.unlink()

    # 2. Archive the remaining contents to the main JAR.
    archive_directory_contents(
        archive_path=main_archive_file,
        source_dir=platform_path,
        description=f"Archiving all contents to main JAR '{main_archive_file.name}'"
    )
    print(f"Finished platform: {platform_path}")
    print("--------------------------------")


# --- Main Execution ---

def main():
    """Main script entry point."""
    # --- OS Check ---
    if sys.platform != "win32":
        print("Error: This script is intended to be run on Windows.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Package ONNX Runtime Java artifacts."
    )
    parser.add_argument(
        "--package_type",
        type=str,
        choices=['cpu', 'gpu'],
        default='cpu',
        help="The type of package to build ('cpu' or 'gpu')."
    )
    args = parser.parse_args()

    # --- Configuration ---
    try:
        # Base directory where all unzipped artifacts are.
        build_dir = os.environ['BUILD_BINARIESDIRECTORY']
        artifacts_base_dir = Path(build_dir) / "java-artifact"
    except KeyError:
        print("Error: Environment variable BUILD_BINARIESDIRECTORY is not set.", file=sys.stderr)
        sys.exit(1)

    # Main Windows CPU package directory, used as the destination for all archives
    # and for discovering the version.
    primary_package_dir = artifacts_base_dir / "onnxruntime-java-win-x64"

    if not primary_package_dir.is_dir():
        print(f"Error: Primary package directory not found at '{primary_package_dir}'", file=sys.stderr)
        sys.exit(1)

    # --- Version Discovery ---
    print(f"Discovering version from JAR files in '{primary_package_dir}'...")
    jar_pattern = str(primary_package_dir / "onnxruntime*-*.jar")
    jar_files = [
        Path(f) for f in glob.glob(jar_pattern)
        if "-sources.jar" not in f and "-javadocs.jar" not in f
    ]

    if not jar_files:
        print(f"Error: Could not find a main JAR file in '{primary_package_dir}' to determine the version.", file=sys.stderr)
        sys.exit(1)

    main_cpu_jar_file = jar_files[0]
    main_cpu_jar_file_stem = main_cpu_jar_file.stem
    try:
        if main_cpu_jar_file_stem.startswith("onnxruntime_gpu-"):
            version = main_cpu_jar_file_stem.replace("onnxruntime_gpu-", "", 1)
        elif main_cpu_jar_file_stem.startswith("onnxruntime-"):
            version = main_cpu_jar_file_stem.replace("onnxruntime-", "", 1)
        else:
            raise ValueError(f"Unexpected JAR file name format: {main_cpu_jar_file.name}")
    except ValueError as e:
        print(f"Error: Failed to parse version from filename '{main_cpu_jar_file.name}'. Reason: {e}", file=sys.stderr)
        sys.exit(1)


    print(f"Version discovered: {version}")

    # --- Package Definitions ---
    # Defines the platforms and libraries for each package type (cpu/gpu).
    package_definitions: Dict[str, Dict[str, Any]] = {
        'cpu': {
            'package_name': 'onnxruntime',
            'platforms': [
                {'path': 'onnxruntime-java-linux-x64', 'lib': 'libcustom_op_library.so', 'archive_lib': True},
                {'path': 'onnxruntime-java-osx-x86_64', 'lib': 'libcustom_op_library.dylib', 'archive_lib': True},
                {'path': 'onnxruntime-java-linux-aarch64', 'lib': 'libcustom_op_library.so', 'archive_lib': False},
                {'path': 'onnxruntime-java-osx-arm64', 'lib': 'libcustom_op_library.dylib', 'archive_lib': False}
            ]
        },
        'gpu': {
            'package_name': 'onnxruntime_gpu',
            'platforms': [
                {
                    'path': 'onnxruntime-java-linux-x64-tensorrt',
                    'lib': 'libcustom_op_library.so',
                    'archive_lib': False,
                    'gpu_libs': [
                        {'source': artifacts_base_dir / 'onnxruntime-java-linux-x64/ai/onnxruntime/native/linux-x64/libonnxruntime_providers_cuda.so', 'dest': 'ai/onnxruntime/native/linux-x64'},
                        {'source': artifacts_base_dir / 'onnxruntime-java-linux-x64/ai/onnxruntime/native/linux-x64/libonnxruntime_providers_tensorrt.so', 'dest': 'ai/onnxruntime/native/linux-x64'}
                    ]
                },
                {
                    'path': 'onnxruntime-java-win-x64-gpu',
                    'lib': 'custom_op_library.dll',
                    'archive_lib': False,
                    'gpu_libs': [
                        {'source': artifacts_base_dir / 'onnxruntime-java-win-x64/ai/onnxruntime/native/win-x64/onnxruntime_providers_cuda.dll', 'dest': 'ai/onnxruntime/native/win-x64'},
                        {'source': artifacts_base_dir / 'onnxruntime-java-win-x64/ai/onnxruntime/native/win-x64/onnxruntime_providers_tensorrt.dll', 'dest': 'ai/onnxruntime/native/win-x64'}
                    ]
                }
            ]
        }
    }


    # --- Processing Loop ---
    package = package_definitions[args.package_type]
    package_name = package['package_name']
    print(f"\n## Configuring for {args.package_type.upper()} package build...")
    print(f"## Processing Package: {package_name}")

    # Define final archive paths. All archives are placed in the primary directory.
    main_archive_file = primary_package_dir / f"{package_name}-{version}.jar"
    test_archive_file = primary_package_dir / f"testing-{package_name}.jar"

    # Clean up any old archive files before starting to avoid appending to them
    main_archive_file.unlink(missing_ok=True)
    test_archive_file.unlink(missing_ok=True)

    for platform in package['platforms']:
        platform_full_path = artifacts_base_dir / platform['path']

        # --- GPU Pre-processing Step ---
        if 'gpu_libs' in platform:
            for gpu_lib in platform['gpu_libs']:
                dest_dir = platform_full_path / gpu_lib['dest']
                source_file = Path(gpu_lib['source'])
                print(f"  -> Copying GPU library '{source_file.name}' to '{dest_dir}'")

                if not source_file.is_file():
                    print(f"Error: GPU source library not found: {source_file}", file=sys.stderr)
                    sys.exit(1)

                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, dest_dir)

        # Call the main processing function
        process_platform_archive(
            platform_path=platform_full_path,
            main_archive_file=main_archive_file,
            test_archive_file=test_archive_file,
            custom_lib_file=platform['lib'],
            archive_custom_lib=platform['archive_lib']
        )
    
    print("\nScript completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}", file=sys.stderr)
        sys.exit(1)