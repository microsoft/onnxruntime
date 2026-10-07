#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Install and consume a minimal static package with the 1DS SDK's link interfaces.

The fixture isolates ORT's package configuration from unrelated dependencies. Android
uses the NDK when ANDROID_NDK_HOME is set; host builds exercise the Linux export graph.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "cmake" / "onnxruntime_telemetry_config.cmake"

_PRODUCER = """
cmake_minimum_required(VERSION 3.28)
project(telemetry_package C)
include(GNUInstallDirs)
find_package(Threads REQUIRED)
add_library(mat STATIC mat.c)
target_link_libraries(mat PUBLIC Threads::Threads)
add_library(onnxruntime_common STATIC common.c)
target_link_libraries(onnxruntime_common PRIVATE mat)
set(export_targets onnxruntime_common mat)
if(TEST_PLATFORM STREQUAL "Linux")
  add_library(libcurl_static STATIC curl.c)
  target_link_libraries(libcurl_static PUBLIC Threads::Threads)
  target_compile_definitions(mat PRIVATE USE_CURL)
  target_link_libraries(mat PRIVATE
    "$<BUILD_INTERFACE:libcurl_static>"
    "$<INSTALL_INTERFACE:MSTelemetry::curl_dependency>")
  list(APPEND export_targets libcurl_static)
endif()
set(onnxruntime_USE_1DS_TELEMETRY ON)
set(onnxruntime_BUILD_SHARED_LIB OFF)
set(PROJECT_CONFIG_CONTENT "include(CMakeFindDependencyMacro)\\n")
# Select the package dependency graph without changing the host compiler/toolchain.
function(append_telemetry_config)
  set(WIN32 FALSE)
  set(APPLE FALSE)
  set(CMAKE_SYSTEM_NAME "${TEST_PLATFORM}")
  include("${TELEMETRY_CONFIG}")
  set(PROJECT_CONFIG_CONTENT "${PROJECT_CONFIG_CONTENT}" PARENT_SCOPE)
endfunction()
append_telemetry_config()
string(APPEND PROJECT_CONFIG_CONTENT
  "include(\\"\\${CMAKE_CURRENT_LIST_DIR}/onnxruntimeTargets.cmake\\")\\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntimeConfig.cmake" "${PROJECT_CONFIG_CONTENT}")
install(TARGETS ${export_targets} EXPORT onnxruntimeTargets ARCHIVE DESTINATION lib)
install(EXPORT onnxruntimeTargets NAMESPACE onnxruntime:: DESTINATION lib/cmake/onnxruntime)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/onnxruntimeConfig.cmake" DESTINATION lib/cmake/onnxruntime)
"""

_CONSUMER = """
cmake_minimum_required(VERSION 3.28)
project(telemetry_consumer C)
if(PREEXISTING_CURL)
  add_library(MSTelemetry::curl_dependency INTERFACE IMPORTED)
  set_property(TARGET MSTelemetry::curl_dependency PROPERTY
    INTERFACE_LINK_LIBRARIES onnxruntime::libcurl_static)
  set_property(TARGET MSTelemetry::curl_dependency PROPERTY TEST_PRESERVED TRUE)
endif()
find_package(onnxruntime CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
if(NOT TARGET Threads::Threads)
  message(FATAL_ERROR "The installed SDK requires Threads::Threads")
endif()
if(TEST_PLATFORM STREQUAL "Linux")
  get_target_property(curl_link MSTelemetry::curl_dependency INTERFACE_LINK_LIBRARIES)
  if(NOT curl_link STREQUAL "onnxruntime::libcurl_static")
    message(FATAL_ERROR "The installed SDK must use the exported curl archive")
  endif()
  if(PREEXISTING_CURL)
    get_target_property(preserved MSTelemetry::curl_dependency TEST_PRESERVED)
    if(NOT preserved)
      message(FATAL_ERROR "A caller's existing dependency target was replaced")
    endif()
  endif()
elseif(TARGET MSTelemetry::curl_dependency)
  message(FATAL_ERROR "The Android Java transport must not require curl")
endif()
add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE onnxruntime::onnxruntime_common)
"""


class TelemetryPackageTest(unittest.TestCase):
    def test_source_sdk_selection_with_and_without_vcpkg(self):
        external = (_ROOT / "cmake" / "external" / "onnxruntime_external_deps.cmake").read_text(encoding="utf-8")
        telemetry = external[external.index("# 1DS SDK (cpp_client_telemetry)") :]
        telemetry = telemetry[: telemetry.index("FILE(TO_NATIVE_PATH")]
        dependency = next(
            line
            for line in (_ROOT / "cmake" / "deps.txt").read_text(encoding="utf-8").splitlines()
            if line.startswith("cpp_client_telemetry;")
        )
        _, url, sha1 = dependency.split(";")
        with tempfile.TemporaryDirectory(prefix="ort-telemetry-source-") as temporary:
            root = Path(temporary)
            (root / "telemetry.cmake").write_text(telemetry, encoding="utf-8")
            (root / "mat.c").write_text("int mat_value(void) { return 42; }\n", encoding="utf-8")
            (root / "CMakeLists.txt").write_text(
                textwrap.dedent(
                    """
                    cmake_minimum_required(VERSION 3.28)
                    project(telemetry_source C)
                    add_library(mat STATIC mat.c)
                    function(onnxruntime_fetchcontent_declare name)
                      cmake_parse_arguments(SDK "" "URL;URL_HASH" "" ${ARGN})
                      if(NOT name STREQUAL "cpp_client_telemetry" OR
                         NOT SDK_URL STREQUAL DEP_URL_cpp_client_telemetry OR
                         NOT SDK_URL_HASH STREQUAL "SHA1=${DEP_SHA1_cpp_client_telemetry}")
                        message(FATAL_ERROR "The SDK must use the verified GitHub source pin")
                      endif()
                    endfunction()
                    macro(onnxruntime_fetchcontent_makeavailable name)
                      if(BUILD_SHARED_LIBS)
                        message(FATAL_ERROR "The source SDK must be embedded as a static library")
                      endif()
                      if(NOT BUILD_VERSION STREQUAL expected_sdk_version)
                        message(FATAL_ERROR "SDK version metadata must match the source pin")
                      endif()
                      set_property(GLOBAL PROPERTY sdk_source_selected TRUE)
                    endmacro()
                    function(check_source platform use_vcpkg expected_curl)
                      set(onnxruntime_USE_1DS_TELEMETRY ON)
                      set(onnxruntime_BUILD_SHARED_LIB ON)
                      set(BUILD_SHARED_LIBS ON)
                      set(BUILD_VERSION caller-version)
                      set(onnxruntime_USE_VCPKG "${use_vcpkg}")
                      set(CMAKE_SYSTEM_NAME "${platform}")
                      set(WIN32 FALSE)
                      set(APPLE FALSE)
                      set(ANDROID FALSE)
                      set_property(GLOBAL PROPERTY sdk_source_selected FALSE)
                      if(platform STREQUAL "Windows")
                        set(WIN32 TRUE)
                      elseif(platform STREQUAL "Darwin")
                        set(APPLE TRUE)
                      endif()
                      include("${CMAKE_CURRENT_SOURCE_DIR}/telemetry.cmake")
                      get_property(sdk_source_selected GLOBAL PROPERTY sdk_source_selected)
                      if(NOT sdk_source_selected OR NOT BUILD_SHARED_LIBS OR
                         NOT BUILD_VERSION STREQUAL "caller-version")
                        message(FATAL_ERROR "SDK source selection or shared-library restoration failed")
                      endif()
                      if(NOT MATSDK_CURL_PROVIDER STREQUAL expected_curl)
                        message(FATAL_ERROR "Unexpected curl provider for ${platform}/${use_vcpkg}")
                      endif()
                    endfunction()
                    check_source(Windows OFF SYSTEM)
                    check_source(Windows ON SYSTEM)
                    check_source(Linux OFF FETCH)
                    check_source(Linux ON SYSTEM)
                    check_source(Darwin OFF SYSTEM)
                    check_source(Darwin ON SYSTEM)
                    """
                ),
                encoding="utf-8",
            )
            self._run(
                "cmake",
                "-S",
                str(root),
                "-B",
                str(root / "build"),
                f"-DDEP_URL_cpp_client_telemetry={url}",
                f"-DDEP_SHA1_cpp_client_telemetry={sha1}",
                f"-Dexpected_sdk_version={url.rsplit('/v', 1)[-1].removesuffix('.zip')}",
                *(["-G", "Visual Studio 18 2026", "-A", "x64"] if sys.platform == "win32" else []),
            )

    def test_vcpkg_telemetry_feature_installs_transport_not_sdk(self):
        manifest = json.loads((_ROOT / "cmake" / "vcpkg.json").read_text(encoding="utf-8"))
        dependencies = manifest["features"]["telemetry"]["dependencies"]
        self.assertEqual({dependency["name"] for dependency in dependencies}, {"curl", "mbedtls"})
        self.assertTrue(all(dependency["platform"] == "linux" for dependency in dependencies))

    def test_production_include_without_module_search_path(self):
        includes = [
            line.strip()
            for line in (_ROOT / "cmake" / "CMakeLists.txt").read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("include(") and "onnxruntime_telemetry_config" in line
        ]
        self.assertEqual(len(includes), 1)
        with tempfile.TemporaryDirectory(prefix="ort-telemetry-include-") as temporary:
            root = Path(temporary)
            shutil.copyfile(_CONFIG, root / _CONFIG.name)
            script = root / "include.cmake"
            script.write_text(
                'set(CMAKE_MODULE_PATH "")\n'
                "set(onnxruntime_USE_1DS_TELEMETRY ON)\n"
                "set(WIN32 FALSE)\nset(APPLE FALSE)\nset(CMAKE_SYSTEM_NAME Android)\n"
                + includes[0]
                + '\nif(NOT PROJECT_CONFIG_CONTENT STREQUAL "find_dependency(Threads)\\n")\n'
                '  message(FATAL_ERROR "The installed SDK dependency was not added")\nendif()\n',
                encoding="utf-8",
            )
            self._run("cmake", "-P", str(script))

    def _run(self, *command):
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)
        self.assertEqual(result.returncode, 0, f"{command}\n{result.stdout}\n{result.stderr}")

    def _install_and_consume(self, platform_name, *, android=False, preexisting_curl=False):
        with tempfile.TemporaryDirectory(prefix="ort-telemetry-package-") as temporary:
            root = Path(temporary)
            producer = root / "producer"
            consumer = root / "consumer"
            producer.mkdir()
            consumer.mkdir()
            (producer / "CMakeLists.txt").write_text(textwrap.dedent(_PRODUCER), encoding="utf-8")
            (producer / "common.c").write_text(
                "int mat_value(void);\nint ort_value(void) { return mat_value(); }\n", encoding="utf-8"
            )
            (producer / "mat.c").write_text(
                "#ifdef USE_CURL\nint curl_value(void);\n"
                "int mat_value(void) { return curl_value(); }\n"
                "#else\nint mat_value(void) { return 42; }\n#endif\n",
                encoding="utf-8",
            )
            (producer / "curl.c").write_text("int curl_value(void) { return 42; }\n", encoding="utf-8")
            (consumer / "CMakeLists.txt").write_text(textwrap.dedent(_CONSUMER), encoding="utf-8")
            (consumer / "main.c").write_text(
                "int ort_value(void);\nint main(void) { return ort_value() == 42 ? 0 : 1; }\n", encoding="utf-8"
            )

            configure = [f"-DTEST_PLATFORM={platform_name}", "-DCMAKE_BUILD_TYPE=Release"]
            if android:
                ndk = Path(os.environ["ANDROID_NDK_HOME"])
                configure += [
                    "-G",
                    "Ninja",
                    f"-DCMAKE_TOOLCHAIN_FILE={ndk / 'build/cmake/android.toolchain.cmake'}",
                    "-DANDROID_ABI=arm64-v8a",
                    "-DANDROID_PLATFORM=android-28",
                ]
                ninja = shutil.which("ninja")
                self.assertIsNotNone(ninja, "Ninja must be on PATH for the Android cross-link")
                configure.append(f"-DCMAKE_MAKE_PROGRAM={ninja}")
            elif sys.platform == "win32":
                configure += ["-G", "Visual Studio 18 2026", "-A", "x64"]

            install = root / "install"
            self._run(
                "cmake",
                "-S",
                str(producer),
                "-B",
                str(root / "producer-build"),
                f"-DTELEMETRY_CONFIG={_CONFIG}",
                f"-DCMAKE_INSTALL_PREFIX={install}",
                *configure,
            )
            self._run("cmake", "--build", str(root / "producer-build"), "--config", "Release")
            self._run("cmake", "--install", str(root / "producer-build"), "--config", "Release")
            self._run(
                "cmake",
                "-S",
                str(consumer),
                "-B",
                str(root / "consumer-build"),
                f"-DCMAKE_PREFIX_PATH={install}",
                f"-DCMAKE_FIND_ROOT_PATH={install}",
                f"-DPREEXISTING_CURL={'ON' if preexisting_curl else 'OFF'}",
                *configure,
            )
            self._run("cmake", "--build", str(root / "consumer-build"), "--config", "Release")
            if not android:
                executable = root / "consumer-build"
                executable /= "Release/consumer.exe" if sys.platform == "win32" else "consumer"
                self._run(str(executable))

    def test_bundled_linux_static_install_and_consume(self):
        for preexisting_curl in (False, True):
            with self.subTest(preexisting_curl=preexisting_curl):
                self._install_and_consume("Linux", preexisting_curl=preexisting_curl)

    def test_bundled_android_java_static_install_and_consume(self):
        self._install_and_consume("Android")

    @unittest.skipUnless(os.environ.get("ANDROID_NDK_HOME"), "Set ANDROID_NDK_HOME for an Android cross-link")
    def test_bundled_android_ndk_static_install_and_consume(self):
        self._install_and_consume("Android", android=True)


if __name__ == "__main__":
    unittest.main()
