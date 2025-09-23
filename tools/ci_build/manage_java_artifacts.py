# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script runs after ORT jars are built. It picks up the jars from ORT's build dir then repack them a bit.

import argparse
import logging
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# --- Helper Functions ---
def run_command(command: list, working_dir: Path):
    """Runs a command in a specified directory and checks for errors."""
    logging.info(f"Running command: '{' '.join(map(str, command))}' in '{working_dir}'")
    try:
        # On Windows, shell=True is required to correctly locate and execute .bat or .cmd files
        # like gradlew.bat and mvn.cmd that may be in the system's PATH.
        use_shell = sys.platform == "win32"
        subprocess.run(command, cwd=working_dir, check=True, shell=use_shell)
        logging.info("Command successful.")
    except subprocess.CalledProcessError as e:
        # Output will have been streamed, so we just need to log the failure.
        logging.error(f"Command failed with exit code {e.returncode}")
        raise
    except FileNotFoundError:
        logging.error(
            f"Command failed: The executable '{command[0]}' was not found. "
            "Please ensure it is installed and that its location is in the system's PATH environment variable."
        )
        raise


def log_directory_contents(dir_path: Path, description: str):
    """Logs the contents of a directory for debugging."""
    logging.info(f"--- Listing contents of {description} at '{dir_path}' ---")
    if not dir_path.is_dir():
        logging.warning(f"Directory does not exist: {dir_path}")
        return
    contents = list(dir_path.rglob("*"))
    if not contents:
        logging.warning(f"Directory is empty: {dir_path}")
    else:
        for item in contents:
            logging.info(f"  - {item.relative_to(dir_path)}")
    logging.info("--- End of directory listing ---")


def create_zip_from_directory(zip_file_path: Path, source_dir: Path):
    """Creates a zip file from the contents of a source directory."""
    logging.info(f"Creating archive '{zip_file_path}' from directory '{source_dir}'...")
    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in source_dir.walk():
            for file in files:
                file_path = root / file
                archive_name = file_path.relative_to(source_dir)
                zipf.write(file_path, archive_name)
    logging.info("Archive created successfully.")


# --- New function for validation ---
def validate_artifacts(
    platform_dir: Path, main_jar: Path, main_pom: Path, testing_jar: Path, version: str, artifact_id: str
):
    """Uses Maven to validate the generated JAR and POM files."""
    logging.info("--- Starting Maven Artifact Validation ---")
    maven_executable = "mvn.cmd" if sys.platform == "win32" else "mvn"
    group_id = "com.microsoft.onnxruntime"  # Assuming this is constant

    # 1. Validate the main ONNX Runtime JAR and its POM
    logging.info(f"Validating main artifact: {main_jar.name}")
    install_main_cmd = [
        maven_executable,
        "install:install-file",
        f"-Dfile={main_jar.resolve()}",
        f"-DpomFile={main_pom.resolve()}",
        # Adding these makes the command more robust and less prone to errors
        f"-DgroupId={group_id}",
        f"-DartifactId={artifact_id}",
        f"-Dversion={version}",
        "-Dpackaging=jar",
    ]
    run_command(install_main_cmd, working_dir=platform_dir)
    logging.info("Main artifact validated successfully.")

    # 2. Validate the testing JAR (it has no POM, so we supply all info)
    logging.info(f"Validating testing artifact: {testing_jar.name}")
    install_testing_cmd = [
        maven_executable,
        "install:install-file",
        f"-Dfile={testing_jar.resolve()}",
        f"-DgroupId={group_id}",
        f"-DartifactId={artifact_id}-testing",
        f"-Dversion={version}",
        "-Dpackaging=jar",
    ]
    run_command(install_testing_cmd, working_dir=platform_dir)
    logging.info("Testing artifact validated successfully.")
    logging.info("--- Maven Artifact Validation Complete ---")


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(description="Builds and packages Java artifacts, PDBs, and notice files.")
    parser.add_argument("--sources-dir", required=True, type=Path, help="Path to the build sources directory.")
    parser.add_argument("--binaries-dir", required=True, type=Path, help="Path to the build binaries directory.")
    parser.add_argument("--platform", required=True, help="Platform string (e.g., x64).")
    parser.add_argument(
        "--java-artifact-id", required=True, help="The Java artifact ID (e.g., onnxruntime or onnxruntime_gpu)."
    )
    parser.add_argument(
        "--build-config",
        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
        default="RelWithDebInfo",
        help="The CMake build configuration type.",
    )
    parser.add_argument(
        "--pre-release-version-suffix-string",
        choices=["alpha", "beta", "rc", "none"],
        default="none",
        help="The pre-release version suffix string.",
    )
    parser.add_argument(
        "--pre-release-version-suffix-number", type=int, default=0, help="The pre-release version suffix number."
    )
    parser.add_argument("--commit-hash", required=True, help="The git commit hash.")
    parser.add_argument("--build-only", action="store_true", help="Flag to indicate if this is a build-only run.")
    args = parser.parse_args()

    # --- 1. Version and Build Logic ---
    # Determine the repository root from the script's location
    repo_root = Path(__file__).resolve().parent.parent.parent
    version_file_path = repo_root / "VERSION_NUMBER"

    logging.info(f"Reading base version from {version_file_path}")
    if not version_file_path.is_file():
        raise FileNotFoundError(f"Version file not found at {version_file_path}")

    base_version = version_file_path.read_text(encoding="utf-8").strip()

    # Validate the version format
    if not re.match(r"^\d+\.\d+\.\d+$", base_version):
        raise ValueError(f"Version '{base_version}' from {version_file_path} is not in the required x.y.z format.")

    logging.info(f"Successfully read and validated base version: {base_version}")

    # Start with the base version and conditionally append the pre-release suffix.
    full_version = base_version
    if args.pre_release_version_suffix_string != "none":
        if args.pre_release_version_suffix_number <= 0:
            raise ValueError(
                "Pre-release version suffix number must be a positive integer if a suffix string is provided."
            )
        # Append the suffix, conforming to Maven standards (e.g., 1.2.3-rc1)
        full_version += f"-{args.pre_release_version_suffix_string}{args.pre_release_version_suffix_number}"

    logging.info(f"Using full version: {full_version}")

    # Use the java subdirectory of the repository root as the working directory for Gradle
    java_working_dir = repo_root / "java"

    build_config_dir = args.binaries_dir / args.build_config
    cmake_build_dir_arg = f"-DcmakeBuildDir={build_config_dir}"
    version_property_arg = f"-Dorg.gradle.project.version={full_version}"

    # Construct the absolute path to the Gradle wrapper
    gradle_executable_name = "gradlew.bat" if sys.platform == "win32" else "gradlew"
    gradle_executable_path = java_working_dir / gradle_executable_name

    # Rebuild the jar so that we can change the version
    gradle_args = [cmake_build_dir_arg, version_property_arg]
    if args.java_artifact_id == "onnxruntime_gpu":
        gradle_args.append("-DUSE_CUDA")
        gradle_args.append("-DUSE_TENSORRT")
    run_command([str(gradle_executable_path), "cmakeBuild", *gradle_args], working_dir=java_working_dir)
    if args.build_only:
        run_command(
            [
                str(gradle_executable_path),
                "testClasses",
                "--warning-mode",
                "all",
                *gradle_args,
            ],
            working_dir=java_working_dir,
        )
    else:
        run_command(
            [
                str(gradle_executable_path),
                "cmakeCheck",
                "--warning-mode",
                "all",
                *gradle_args,
            ],
            working_dir=java_working_dir,
        )

    # --- 2. Path Definitions ---
    platform_dir = args.binaries_dir / f"onnxruntime-java-win-{args.platform}"
    stage_dir = platform_dir / "stage"
    native_folder = stage_dir / "ai" / "onnxruntime" / "native" / f"win-{args.platform}"
    main_jar_name = f"{args.java_artifact_id}-{full_version}.jar"
    main_jar_path = platform_dir / main_jar_name
    final_pom_path = platform_dir / f"{args.java_artifact_id}-{full_version}.pom"
    testing_jar_path = platform_dir / "testing.jar"

    # --- 3. Packaging Logic ---
    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
        native_folder.mkdir(parents=True, exist_ok=True)

        gradle_libs_dir = java_working_dir / "build" / "libs"
        log_directory_contents(gradle_libs_dir, "Gradle build output libs")

        # FIX: Filter glob results to find the main artifact JAR, excluding sources and javadoc.
        main_jars = [
            p
            for p in gradle_libs_dir.glob("*.jar")
            if not p.name.endswith("-sources.jar") and not p.name.endswith("-javadoc.jar")
        ]

        if not main_jars:
            raise FileNotFoundError(f"Gradle build finished, but no main artifact JAR was found in {gradle_libs_dir}")
        if len(main_jars) > 1:
            logging.warning(f"Found multiple potential main JARs: {[p.name for p in main_jars]}. Using the first one.")

        source_jar_path = main_jars[0]
        logging.info(f"Found source JAR to copy: {source_jar_path.name}")

        # The main JAR file is copied to its final name directly.
        shutil.copy2(source_jar_path, main_jar_path)

        # Now, find and copy the associated sources and javadoc JARs, renaming them to match.
        source_basename = source_jar_path.stem  # e.g., 'onnxruntime-1.23.0'
        dest_basename = main_jar_path.stem  # e.g., 'onnxruntime_gpu-1.23.0'

        for classifier in ["sources", "javadoc"]:
            source_classified_jar = gradle_libs_dir / f"{source_basename}-{classifier}.jar"
            if source_classified_jar.is_file():
                dest_classified_jar = platform_dir / f"{dest_basename}-{classifier}.jar"
                logging.info(f"Copying classified artifact: {source_classified_jar.name} -> {dest_classified_jar.name}")
                shutil.copy2(source_classified_jar, dest_classified_jar)
            else:
                logging.warning(f"Optional artifact '{source_classified_jar.name}' not found, skipping.")

        log_directory_contents(platform_dir, "final platform directory before JAR processing")

        pom_archive_path = f"META-INF/maven/com.microsoft.onnxruntime/{args.java_artifact_id}/pom.xml"
        with zipfile.ZipFile(main_jar_path, "r") as jar:
            jar.extract(pom_archive_path, path=platform_dir)

        shutil.move(str(platform_dir / pom_archive_path), str(final_pom_path))
        shutil.rmtree(platform_dir / "META-INF")

        shutil.copy2(args.sources_dir / "docs" / "Privacy.md", stage_dir)
        shutil.copy2(args.sources_dir / "ThirdPartyNotices.txt", stage_dir)
        (stage_dir / "GIT_COMMIT_ID").write_text(args.commit_hash, encoding="utf-8")

        with zipfile.ZipFile(main_jar_path, "a") as jar:
            for root, _, files in stage_dir.walk():
                for file in files:
                    file_path = root / file
                    jar.write(file_path, file_path.relative_to(stage_dir))

        test_classes_dir = args.sources_dir / "java" / "build" / "classes" / "java" / "test"
        test_resources_dir = args.sources_dir / "java" / "build" / "resources" / "test"

        create_zip_from_directory(testing_jar_path, test_classes_dir)

        native_resource_path = test_resources_dir / "ai" / "onnxruntime" / "native"
        if native_resource_path.exists():
            shutil.rmtree(native_resource_path)

        with zipfile.ZipFile(testing_jar_path, "a") as jar:
            for root, _, files in test_resources_dir.walk():
                for file in files:
                    file_path = root / file
                    jar.write(file_path, file_path.relative_to(test_resources_dir))

        logging.info("Java artifact packaging complete.")

        # --- 4. Validation Step ---
        validate_artifacts(
            platform_dir=platform_dir,
            main_jar=main_jar_path,
            main_pom=final_pom_path,
            testing_jar=testing_jar_path,
            version=full_version,
            artifact_id=args.java_artifact_id,
        )

    finally:
        # 5. Clean up stage directory
        if stage_dir.exists():
            logging.info(f"Cleaning up stage directory: {stage_dir}")
            shutil.rmtree(stage_dir)

    logging.info(f"\nFinal contents of '{platform_dir}':")
    for item in platform_dir.iterdir():
        print(item)


if __name__ == "__main__":
    main()
