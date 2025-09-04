import argparse
import logging
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Helper Functions ---
def run_command(command: list, working_dir: Path):
    """Runs a command in a specified directory and checks for errors."""
    logging.info(f"Running command: '{' '.join(map(str, command))}' in '{working_dir}'")
    try:
        subprocess.run(command, cwd=working_dir, check=True)
        logging.info("Command successful.")
    except subprocess.CalledProcessError as e:
        # Output will have been streamed, so we just need to log the failure.
        logging.error(f"Command failed with exit code {e.returncode}")
        raise

def update_pom_versions(java_source_dir: Path, new_version: str):
    """Finds all pom.xml files and updates their version tag before the build."""
    logging.info(f"Updating all pom.xml files to version '{new_version}'...")
    pom_files = list(java_source_dir.rglob('pom.xml'))
    if not pom_files:
        logging.warning("No pom.xml files found to update.")
        return

    namespace = {'mvn': 'http://maven.apache.org/POM/4.0.0'}
    ET.register_namespace('', namespace['mvn'])

    for pom_file in pom_files:
        try:
            logging.info(f"Processing '{pom_file}'...")
            tree = ET.parse(pom_file)
            root = tree.getroot()
            version_element = root.find('mvn:version', namespace)

            if version_element is not None:
                version_element.text = new_version
                tree.write(pom_file, encoding='utf-8', xml_declaration=True)
            else:
                logging.warning(f"  Could not find <version> tag in '{pom_file}'.")
        except Exception as e:
            logging.error(f"An error occurred while processing '{pom_file}': {e}")


def create_zip_from_directory(zip_file_path: Path, source_dir: Path):
    """Creates a zip file from the contents of a source directory."""
    logging.info(f"Creating archive '{zip_file_path}' from directory '{source_dir}'...")
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in source_dir.walk():
            for file in files:
                file_path = root / file
                archive_name = file_path.relative_to(source_dir)
                zipf.write(file_path, archive_name)
    logging.info("Archive created successfully.")

# --- New function for validation ---
def validate_artifacts(platform_dir: Path, main_jar: Path, main_pom: Path, testing_jar: Path, version: str, artifact_id: str):
    """Uses Maven to validate the generated JAR and POM files."""
    logging.info("--- Starting Maven Artifact Validation ---")
    maven_executable = 'mvn.cmd' if sys.platform == 'win32' else 'mvn'
    group_id = 'com.microsoft.onnxruntime' # Assuming this is constant

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
        "-Dpackaging=jar"
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
        "-Dpackaging=jar"
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
    parser.add_argument("--java-artifact-id", required=True, help="The Java artifact ID.")
    parser.add_argument("--base-version", required=True, help="The base version string for the artifact.")
    parser.add_argument("--version-suffix", default="", help="The version suffix (e.g., -rc.1).")
    parser.add_argument("--commit-hash", required=True, help="The git commit hash.")
    parser.add_argument("--build-only", action="store_true", help="Flag to indicate if this is a build-only run.")
    args = parser.parse_args()

    # --- 1. Version and Build Logic ---
    full_version = f"{args.base_version}{args.version_suffix}"
    logging.info(f"Using full version: {full_version}")

    java_working_dir = args.sources_dir / 'java'
    update_pom_versions(java_working_dir, full_version)

    cmake_build_dir_arg = f"-DcmakeBuildDir={args.binaries_dir / 'RelWithDebInfo'}"
    gradle_executable = 'gradlew.bat' if sys.platform == 'win32' else './gradlew'

    if args.build_only:
        run_command([gradle_executable, "testClasses", cmake_build_dir_arg], working_dir=java_working_dir)

    run_command([gradle_executable, "cmakeCheck", "-x", "test", cmake_build_dir_arg, "--warning-mode", "all"],
                working_dir=java_working_dir)

    # --- 2. Path Definitions ---
    platform_dir = args.binaries_dir / f"onnxruntime-java-win-{args.platform}"
    stage_dir = platform_dir / "stage"
    native_folder = stage_dir / "ai" / "onnxruntime" / "native" / f"win-{args.platform}"
    main_jar_name = f"onnxruntime-{full_version}.jar"
    main_jar_path = platform_dir / main_jar_name
    final_pom_path = platform_dir / f"onnxruntime-{full_version}.pom"
    testing_jar_path = platform_dir / "testing.jar"

    # --- 3. Packaging Logic ---
    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
        native_folder.mkdir(parents=True, exist_ok=True)

        source_jar_path = next((args.binaries_dir / "RelWithDebInfo" / "java" / "build" / "libs").glob("*.jar"))
        shutil.copy2(source_jar_path, platform_dir)

        pom_archive_path = f"META-INF/maven/com.microsoft.onnxruntime/{args.java_artifact_id}/pom.xml"
        with zipfile.ZipFile(main_jar_path, 'r') as jar:
            jar.extract(pom_archive_path, path=platform_dir)
        
        (platform_dir / pom_archive_path).rename(final_pom_path)
        shutil.rmtree(platform_dir / "META-INF")

        shutil.copy2(args.binaries_dir / "RelWithDebInfo" / "onnxruntime.pdb", native_folder)
        shutil.copy2(args.binaries_dir / "RelWithDebInfo" / "onnxruntime4j_jni.pdb", native_folder)
        shutil.copy2(args.sources_dir / "docs" / "Privacy.md", stage_dir)
        shutil.copy2(args.sources_dir / "ThirdPartyNotices.txt", stage_dir)
        (stage_dir / "GIT_COMMIT_ID").write_text(args.commit_hash, encoding="utf-8")

        with zipfile.ZipFile(main_jar_path, 'a') as jar:
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
        
        with zipfile.ZipFile(testing_jar_path, 'a') as jar:
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
            artifact_id=args.java_artifact_id
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

