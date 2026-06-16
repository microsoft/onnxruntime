# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import zipfile
from pathlib import Path

import jar_packaging  # The refactored script
import pytest


# Helper to create an empty file
def create_empty_file(path):
    Path(path).touch()


# Helper to create a dummy JAR file
def create_dummy_jar(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n")


@pytest.fixture
def directory_setup_factory(tmp_path):
    """
    A factory fixture that returns a function to set up a test directory
    for a given package type and version.
    """

    def _setup_test_directory(package_type: str, version_string: str):
        """Sets up a temporary directory structure mimicking the build artifacts."""
        java_artifact_dir = tmp_path / "java-artifact"
        win_dir = java_artifact_dir / "onnxruntime-java-win-x64"
        linux_dir = java_artifact_dir / "onnxruntime-java-linux-x64"

        # --- Main artifact directory (Windows) ---
        win_dir.mkdir(parents=True, exist_ok=True)
        artifact_name = f"onnxruntime_{package_type}" if package_type == "gpu" else "onnxruntime"
        create_dummy_jar(win_dir / f"{artifact_name}-{version_string}.jar")
        create_dummy_jar(win_dir / f"{artifact_name}-{version_string}-sources.jar")
        create_dummy_jar(win_dir / f"{artifact_name}-{version_string}-javadoc.jar")
        create_empty_file(win_dir / f"{artifact_name}-{version_string}.pom")
        create_dummy_jar(win_dir / "testing.jar")
        (win_dir / "_manifest" / "spdx_2.2").mkdir(parents=True, exist_ok=True)

        # --- Linux platform ---
        linux_native_dir = linux_dir / "ai" / "onnxruntime" / "native" / "linux-x64"
        linux_native_dir.mkdir(parents=True, exist_ok=True)
        create_empty_file(linux_dir / "libcustom_op_library.so")
        create_empty_file(linux_native_dir / "libonnxruntime.so")
        create_empty_file(linux_native_dir / "libonnxruntime4j_jni.so")
        if package_type == "gpu":
            create_empty_file(linux_native_dir / "libonnxruntime_providers_cuda.so")
        (linux_dir / "_manifest" / "spdx_2.2").mkdir(parents=True, exist_ok=True)

        # --- macOS and other platforms (for CPU test) ---
        if package_type == "cpu":
            # Add linux-aarch64 and osx-arm64 for CPU test
            linux_aarch64_dir = java_artifact_dir / "onnxruntime-java-linux-aarch64"
            linux_aarch64_native_dir = linux_aarch64_dir / "ai" / "onnxruntime" / "native" / "linux-aarch64"
            linux_aarch64_native_dir.mkdir(parents=True, exist_ok=True)
            create_empty_file(linux_aarch64_dir / "libcustom_op_library.so")

            osx_arm64_dir = java_artifact_dir / "onnxruntime-java-osx-arm64"
            osx_arm64_native_dir = osx_arm64_dir / "ai" / "onnxruntime" / "native" / "osx-arm64"
            osx_arm64_native_dir.mkdir(parents=True, exist_ok=True)
            create_empty_file(osx_arm64_dir / "libcustom_op_library.dylib")

        return tmp_path

    return _setup_test_directory


@pytest.mark.parametrize("version_string", ["1.23.0", "1.23.0-rc1"])
def test_gpu_packaging(directory_setup_factory, version_string):
    """
    Tests the GPU packaging logic for both release and pre-release versions
    to ensure correct files are added to the JARs.
    """
    temp_build_dir = directory_setup_factory("gpu", version_string)

    # Run the packaging script logic
    jar_packaging.run_packaging("gpu", str(temp_build_dir))

    # --- Verification ---
    win_dir = temp_build_dir / "java-artifact" / "onnxruntime-java-win-x64"
    main_jar_path = win_dir / f"onnxruntime_gpu-{version_string}.jar"
    testing_jar_path = win_dir / "testing.jar"

    # 1. Verify the main JAR contains the Linux native libraries
    with zipfile.ZipFile(main_jar_path, "r") as zf:
        jar_contents = zf.namelist()
        assert "ai/onnxruntime/native/linux-x64/libonnxruntime.so" in jar_contents
        assert "ai/onnxruntime/native/linux-x64/libonnxruntime4j_jni.so" in jar_contents
        assert "ai/onnxruntime/native/linux-x64/libonnxruntime_providers_cuda.so" in jar_contents

    # 2. Verify the testing JAR does not contain the custom op library for GPU builds
    with zipfile.ZipFile(testing_jar_path, "r") as zf:
        jar_contents = zf.namelist()
        # The custom op lib for linux is not archived for GPU builds.
        # This checks that it's NOT in the test jar.
        assert "libcustom_op_library.so" not in jar_contents

    # 3. Verify the custom op library was removed from the source linux directory
    linux_dir = temp_build_dir / "java-artifact" / "onnxruntime-java-linux-x64"
    assert not (linux_dir / "libcustom_op_library.so").exists()


@pytest.mark.parametrize("version_string", ["1.23.0", "1.23.0-rc1"])
def test_cpu_packaging(directory_setup_factory, version_string):
    """
    Tests the CPU packaging logic to ensure correct files are added to the JARs.
    """
    temp_build_dir = directory_setup_factory("cpu", version_string)

    # Run the packaging script logic
    jar_packaging.run_packaging("cpu", str(temp_build_dir))

    # --- Verification ---
    win_dir = temp_build_dir / "java-artifact" / "onnxruntime-java-win-x64"
    main_jar_path = win_dir / f"onnxruntime-{version_string}.jar"
    testing_jar_path = win_dir / "testing.jar"

    # 1. Verify the main JAR contains native libraries from all relevant platforms
    with zipfile.ZipFile(main_jar_path, "r") as zf:
        jar_contents = zf.namelist()
        # Linux libs
        assert "ai/onnxruntime/native/linux-x64/libonnxruntime.so" in jar_contents
        assert "ai/onnxruntime/native/linux-x64/libonnxruntime4j_jni.so" in jar_contents
        # GPU libs should NOT be present
        assert "ai/onnxruntime/native/linux-x64/libonnxruntime_providers_cuda.so" not in jar_contents

    # 2. Verify the testing JAR contains the custom op libraries that should be archived
    with zipfile.ZipFile(testing_jar_path, "r") as zf:
        jar_contents = zf.namelist()
        assert "libcustom_op_library.so" in jar_contents
        assert "libcustom_op_library.dylib" in jar_contents

    # 3. Verify the custom op libraries were removed from the source directories
    linux_dir = temp_build_dir / "java-artifact" / "onnxruntime-java-linux-x64"
    linux_aarch64_dir = temp_build_dir / "java-artifact" / "onnxruntime-java-linux-aarch64"
    osx_arm64_dir = temp_build_dir / "java-artifact" / "onnxruntime-java-osx-arm64"
    assert not (linux_dir / "libcustom_op_library.so").exists()
    assert not (linux_aarch64_dir / "libcustom_op_library.so").exists()
    assert not (osx_arm64_dir / "libcustom_op_library.dylib").exists()
