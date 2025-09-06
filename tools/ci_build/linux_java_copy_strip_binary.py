#!/usr/bin/env python3

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
    str_command = ' '.join(map(str, command))
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
    parser.add_argument("-r", "--binary-dir", required=True, type=Path, help="Path to the build binaries directory.")
    parser.add_argument("-a", "--artifact-name", required=True, help="Name of the final artifact directory.")
    parser.add_argument("-c", "--build-config", required=True, help="CMake build configuration (e.g., Release).")
    parser.add_argument("-l", "--lib-name", required=True, help="Filename of the main ONNX Runtime shared library.")
    parser.add_argument("-n", "--native-lib-name", required=True, help="Filename of the JNI shared library.")
    parser.add_argument("-h", "--arch", required=True, help="Architecture string (e.g., osx-x86_64).")
    args = parser.parse_args()

    ## Path Setup and System Info
    ---
    logging.info(f"System Info: {' '.join(platform.uname())}")

    # Define key paths
    source_build_dir = args.binary_dir / args.build_config
    target_artifact_dir = args.binary_dir / args.artifact_name

    # Map architecture names for macOS to align with Java conventions
    arch = args.arch
    if args.lib_name.endswith(".dylib"):
        if arch == 'osx-x86_64':
            arch = 'osx-x64'
            logging.info("Mapped arch from 'osx-x86_64' to 'osx-x64'.")
        elif arch == 'osx-arm64':
            arch = 'osx-aarch64'
            logging.info("Mapped arch from 'osx-arm64' to 'osx-aarch64'.")

    ## Library Processing
    ---
    native_folder = target_artifact_dir / "ai" / "onnxruntime" / "native" / arch
    native_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"Required directories created at {native_folder}")
    logging.info("Copying debug symbols and stripping binaries...")

    if args.lib_name.endswith(".dylib"):  # macOS
        run_command(["dsymutil", source_build_dir / args.lib_name, "-o", native_folder / f"{args.lib_name}.dSYM"])
        shutil.copy2(source_build_dir / args.lib_name, native_folder / "libonnxruntime.dylib")
        run_command(["strip", "-S", native_folder / "libonnxruntime.dylib"])

        run_command(["dsymutil", source_build_dir / args.native_lib_name, "-o", native_folder / f"{args.native_lib_name}.dSYM"])
        shutil.copy2(source_build_dir / args.native_lib_name, native_folder / "libonnxruntime4j_jni.dylib")
        run_command(["strip", "-S", native_folder / "libonnxruntime4j_jni.dylib"])

        shutil.copy2(source_build_dir / "libcustom_op_library.dylib", target_artifact_dir)

    elif args.lib_name.endswith(".so"):  # Linux
        shutil.copy2(source_build_dir / args.lib_name, native_folder / "libonnxruntime.so")
        shutil.copy2(source_build_dir / args.native_lib_name, native_folder / "libonnxruntime4j_jni.so")
        shutil.copy2(source_build_dir / "libcustom_op_library.so", target_artifact_dir)

        for provider in ["cuda", "tensorrt"]:
            provider_lib = source_build_dir / f"libonnxruntime_providers_{provider}.so"
            if provider_lib.exists():
                logging.info(f"Found {provider} provider library. Copying...")
                shared_provider_lib = source_build_dir / "libonnxruntime_providers_shared.so"
                if shared_provider_lib.exists():
                    shutil.copy2(shared_provider_lib, native_folder)
                shutil.copy2(provider_lib, native_folder)
    else:
        logging.warning(f"Unsupported library type for '{args.lib_name}'. No special processing will occur.")

    ## Finalization
    ---
    logging.info(f"--- Final contents of '{target_artifact_dir}' ---")
    for path in sorted(target_artifact_dir.rglob('*')):
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