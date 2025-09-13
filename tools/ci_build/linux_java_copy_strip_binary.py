#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Prepares native shared libraries for the ONNX Runtime Java package.

This script is a build utility that run as part of a packaging pipeline and takes compiled C/C++ shared libraries
(.so, .dylib) and stages them for packaging into a Java JAR file.

It expected the following inputs:
<binary_dir>/
└── <build_config>/
    ├── libonnxruntime.so                (File from --lib-name)
    ├── libonnxruntime4j_jni.so          (File from --native-lib-name)
    ├── libcustom_op_library.so
    │
    ├── (Optional) libonnxruntime_providers_shared.so
    ├── (Optional) libonnxruntime_providers_cuda.so
    └── (Optional) libonnxruntime_providers_tensorrt.so

It performs the following key operations:

1.  Validates the existence of all required source directories and libraries.
2.  Creates the specific Java Native Interface (JNI) directory structure
    (ai/onnxruntime/native/<arch>).
3.  Copies the main, JNI, and custom op libraries to their destinations.
4.  For macOS, extracts debug symbols into .dSYM files using `dsymutil`.
5.  Strips all release binaries of their debug symbols to reduce file size.
6.  Copies optional provider libraries (e.g., CUDA, TensorRT) for Linux builds.

It is intended to be called from a CI/CD pipeline as part of the overall
build process for the onnxruntime-java package.
"""

import argparse
import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# --- Helper Functions ---
def run_command(command: list[str | Path]):
    """Runs an external command and exits the script if the command fails."""
    str_command = " ".join(map(str, command))
    logging.info(f"Running command: '{str_command}'")
    try:
        proc = subprocess.run(command, check=True, text=True, capture_output=True)
        logging.info(f"Successfully executed: {Path(command[0]).name}")
        if proc.stdout:
            logging.debug(f"STDOUT: {proc.stdout.strip()}")
    except FileNotFoundError:
        logging.error(f"Command not found: '{command[0]}'. Please ensure it is installed and in your PATH.")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{Path(e.cmd[0]).name}' failed with exit code {e.returncode}.")
        if e.stdout:
            logging.error(f"STDOUT: {e.stdout.strip()}")
        if e.stderr:
            logging.error(f"STDERR: {e.stderr.strip()}")
        raise


# --- Main Execution ---
def main():
    """Main function to parse arguments and package the native libraries."""
    parser = argparse.ArgumentParser(
        description="Packages ONNX Runtime native libraries for Java.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Arguments
    parser.add_argument("--binary-dir", required=True, type=Path, help="Path to the build binaries directory.")
    parser.add_argument("--artifact-name", required=True, help="Name of the final artifact directory.")
    parser.add_argument("--build-config", required=True, help="CMake build configuration (e.g., Release).")
    parser.add_argument("--lib-name", required=True, help="Filename of the main ONNX Runtime shared library.")
    parser.add_argument("--native-lib-name", required=True, help="Filename of the JNI shared library.")
    parser.add_argument("--arch", required=True, help="Architecture string (e.g., osx-x86_64).")
    args = parser.parse_args()

    # --- Path Setup and Validation ---
    logging.info(f"System Info: {' '.join(platform.uname())}")

    source_build_dir = args.binary_dir / args.build_config
    target_artifact_dir = args.binary_dir / args.artifact_name

    # Validate that the source build directory exists.
    if not source_build_dir.is_dir():
        logging.error(f"Source build directory not found: {source_build_dir}")
        sys.exit(1)

    # Map architecture names for macOS to align with Java conventions
    arch = args.arch
    if args.lib_name.endswith(".dylib"):
        if arch == "osx-x86_64":
            arch = "osx-x64"
        elif arch == "osx-arm64":
            arch = "osx-aarch64"

    # --- Library Processing ---
    native_folder = target_artifact_dir / "ai" / "onnxruntime" / "native" / arch
    native_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"Staging native libraries in: {native_folder}")

    # Validate that all required library files exist before processing.
    main_lib_src = source_build_dir / args.lib_name
    jni_lib_src = source_build_dir / args.native_lib_name

    required_files = [main_lib_src, jni_lib_src]
    lib_suffix = ".dylib" if args.lib_name.endswith(".dylib") else ".so"
    custom_op_lib_src = source_build_dir / f"libcustom_op_library{lib_suffix}"
    required_files.append(custom_op_lib_src)

    for f in required_files:
        if not f.is_file():
            logging.error(f"Required library file not found: {f}")
            sys.exit(1)
    logging.info("All required source library files found.")

    # Start processing now that checks have passed
    if lib_suffix == ".dylib":  # macOS
        logging.info("Processing macOS libraries (.dylib)...")
        run_command(["dsymutil", main_lib_src, "-o", native_folder / f"{args.lib_name}.dSYM"])
        shutil.copy2(main_lib_src, native_folder / "libonnxruntime.dylib")
        run_command(["strip", "-S", native_folder / "libonnxruntime.dylib"])

        run_command(["dsymutil", jni_lib_src, "-o", native_folder / f"{args.native_lib_name}.dSYM"])
        shutil.copy2(jni_lib_src, native_folder / "libonnxruntime4j_jni.dylib")
        run_command(["strip", "-S", native_folder / "libonnxruntime4j_jni.dylib"])

        shutil.copy2(custom_op_lib_src, target_artifact_dir)

    elif lib_suffix == ".so":  # Linux
        logging.info("Processing Linux libraries (.so)...")

        # Main library
        main_lib_dest = native_folder / "libonnxruntime.so"
        shutil.copy2(main_lib_src, main_lib_dest)
        run_command(["strip", "-S", main_lib_dest])

        # JNI library
        jni_lib_dest = native_folder / "libonnxruntime4j_jni.so"
        shutil.copy2(jni_lib_src, jni_lib_dest)
        run_command(["strip", "-S", jni_lib_dest])

        # Custom op library (not stripped as it's for testing)
        shutil.copy2(custom_op_lib_src, target_artifact_dir)

        # Provider checks are optional, so we check for their existence here.
        for provider in ["cuda", "tensorrt"]:
            provider_lib_src = source_build_dir / f"libonnxruntime_providers_{provider}.so"
            if provider_lib_src.exists():
                logging.info(f"Found optional {provider} provider library. Copying and stripping...")

                # Shared provider library
                shared_provider_lib_src = source_build_dir / "libonnxruntime_providers_shared.so"
                if shared_provider_lib_src.exists():
                    shared_provider_dest = native_folder / shared_provider_lib_src.name
                    shutil.copy2(shared_provider_lib_src, shared_provider_dest)
                    run_command(["strip", "-S", shared_provider_dest])

                # Specific provider library
                provider_lib_dest = native_folder / provider_lib_src.name
                shutil.copy2(provider_lib_src, provider_lib_dest)
                run_command(["strip", "-S", provider_lib_dest])
    else:
        logging.warning(f"Unsupported library type for '{args.lib_name}'. No special processing will occur.")

    # --- Finalization ---
    logging.info(f"--- Final contents of '{target_artifact_dir}' ---")
    for path in sorted(target_artifact_dir.rglob("*")):
        logging.info(f"  - {path.relative_to(target_artifact_dir)}")
    logging.info("--- End of contents ---")

    jar_dir_to_remove = target_artifact_dir / "jar"
    if jar_dir_to_remove.is_dir():
        logging.info(f"Removing temporary directory: {jar_dir_to_remove}")
        shutil.rmtree(jar_dir_to_remove)

    logging.info("Script completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script failed due to an unhandled error: {e}")
        sys.exit(1)
