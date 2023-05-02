#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
BUILD_PY = os.path.join(REPO_DIR, "tools", "ci_build", "build.py")
JAVA_ROOT = os.path.join(REPO_DIR, "java")
DEFAULT_BUILD_VARIANT = "Full"

sys.path.insert(0, os.path.join(REPO_DIR, "tools", "python"))
from util import is_windows  # noqa: E402

# We by default will build all 4 ABIs
DEFAULT_BUILD_ABIS = ["armeabi-v7a", "arm64-v8a", "x86", "x86_64"]

# Onnx Runtime native library is built against NDK API 21 by default
# It is possible to build from source for Android API levels below 21, but it is not guaranteed
DEFAULT_ANDROID_MIN_SDK_VER = 21

# Android API 24 is the default target API version for Android builds, based on Microsoft 1CS requirements
# It is possible to build from source using API level 21 and higher as the target SDK version
DEFAULT_ANDROID_TARGET_SDK_VER = 24


def _parse_build_settings(args):
    setting_file = args.build_settings_file.resolve()

    if not setting_file.is_file():
        raise FileNotFoundError(f"Build config file {setting_file} is not a file.")

    with open(setting_file) as f:
        build_settings_data = json.load(f)

    build_settings = {}

    if "build_abis" in build_settings_data:
        build_settings["build_abis"] = build_settings_data["build_abis"]
    else:
        build_settings["build_abis"] = DEFAULT_BUILD_ABIS

    build_params = []
    if "build_params" in build_settings_data:
        build_params += build_settings_data["build_params"]
    else:
        raise ValueError("build_params is required in the build config file")

    if "android_min_sdk_version" in build_settings_data:
        build_settings["android_min_sdk_version"] = build_settings_data["android_min_sdk_version"]
    else:
        build_settings["android_min_sdk_version"] = DEFAULT_ANDROID_MIN_SDK_VER
    build_params += ["--android_api=" + str(build_settings["android_min_sdk_version"])]

    if "android_target_sdk_version" in build_settings_data:
        build_settings["android_target_sdk_version"] = build_settings_data["android_target_sdk_version"]
    else:
        build_settings["android_target_sdk_version"] = DEFAULT_ANDROID_TARGET_SDK_VER

    if build_settings["android_min_sdk_version"] > build_settings["android_target_sdk_version"]:
        raise ValueError(
            "android_min_sdk_version {} cannot be larger than android_target_sdk_version {}".format(
                build_settings["android_min_sdk_version"], build_settings["android_target_sdk_version"]
            )
        )

    build_settings["build_params"] = build_params
    build_settings["build_variant"] = build_settings_data.get("build_variant", DEFAULT_BUILD_VARIANT)

    return build_settings


def _build_aar(args):
    build_settings = _parse_build_settings(args)
    build_dir = os.path.abspath(args.build_dir)
    ops_config_path = os.path.abspath(args.include_ops_by_config) if args.include_ops_by_config else None

    # Setup temp environment for building
    temp_env = os.environ.copy()
    temp_env["ANDROID_HOME"] = os.path.abspath(args.android_sdk_path)
    temp_env["ANDROID_NDK_HOME"] = os.path.abspath(args.android_ndk_path)

    # Temp dirs to hold building results
    intermediates_dir = os.path.join(build_dir, "intermediates")
    build_config = args.config
    aar_dir = os.path.join(intermediates_dir, "aar", build_config)
    jnilibs_dir = os.path.join(intermediates_dir, "jnilibs", build_config)
    exe_dir = os.path.join(intermediates_dir, "executables", build_config)
    base_build_command = [sys.executable, BUILD_PY] + build_settings["build_params"] + ["--config=" + build_config]
    header_files_path = ""
    # Build and install protoc
    protobuf_installation_script = os.path.join(
        REPO_DIR,
        "tools",
        "ci_build",
        "github",
        "linux",
        "docker",
        "inference",
        "x64",
        "python",
        "cpu",
        "scripts",
        "install_protobuf.sh",
    )
    subprocess.run(
        [
            protobuf_installation_script,
            "-p",
            os.path.join(build_dir, "protobuf"),
            "-d",
            os.path.join(REPO_DIR, "cmake", "deps.txt"),
        ],
        shell=False,
        check=True,
    )
    # Build binary for each ABI, one by one
    for abi in build_settings["build_abis"]:
        abi_build_dir = os.path.join(intermediates_dir, abi)
        abi_build_command = [
            *base_build_command,
            "--android_abi=" + abi,
            "--build_dir=" + abi_build_dir,
            "--path_to_protoc_exe",
            os.path.join(build_dir, "protobuf", "bin", "protoc"),
        ]

        if ops_config_path is not None:
            abi_build_command += ["--include_ops_by_config=" + ops_config_path]

        subprocess.run(abi_build_command, env=temp_env, shell=False, check=True, cwd=REPO_DIR)

        # create symbolic links for libonnxruntime.so and libonnxruntime4j_jni.so
        # to jnilibs/[abi] for later compiling the aar package
        abi_jnilibs_dir = os.path.join(jnilibs_dir, abi)
        os.makedirs(abi_jnilibs_dir, exist_ok=True)
        for lib_name in ["libonnxruntime.so", "libonnxruntime4j_jni.so"]:
            target_lib_name = os.path.join(abi_jnilibs_dir, lib_name)
            # If the symbolic already exists, delete it first
            # For some reason, os.path.exists will return false for a symbolic link in Linux,
            # add double check with os.path.islink
            if os.path.exists(target_lib_name) or os.path.islink(target_lib_name):
                os.remove(target_lib_name)
            os.symlink(os.path.join(abi_build_dir, build_config, lib_name), target_lib_name)

        # copy executables for each abi, in case we want to publish those as well
        abi_exe_dir = os.path.join(exe_dir, abi)
        for exe_name in ["libonnxruntime.so", "onnxruntime_perf_test", "onnx_test_runner"]:
            os.makedirs(abi_exe_dir, exist_ok=True)
            target_exe_name = os.path.join(abi_exe_dir, exe_name)
            shutil.copyfile(os.path.join(abi_build_dir, build_config, exe_name), target_exe_name)

        # we only need to define the header files path once
        if not header_files_path:
            header_files_path = os.path.join(abi_build_dir, build_config, "android", "headers")

    # The directory to publish final AAR
    aar_publish_dir = os.path.join(build_dir, "aar_out", build_config)
    os.makedirs(aar_publish_dir, exist_ok=True)

    gradle_path = os.path.join(JAVA_ROOT, "gradlew" if not is_windows() else "gradlew.bat")

    # get the common gradle command args
    gradle_command = [
        gradle_path,
        "--no-daemon",
        "-b=build-android.gradle",
        "-c=settings-android.gradle",
        "-DjniLibsDir=" + jnilibs_dir,
        "-DbuildDir=" + aar_dir,
        "-DheadersDir=" + header_files_path,
        "-DpublishDir=" + aar_publish_dir,
        "-DminSdkVer=" + str(build_settings["android_min_sdk_version"]),
        "-DtargetSdkVer=" + str(build_settings["android_target_sdk_version"]),
        "-DbuildVariant=" + str(build_settings["build_variant"]),
        "-DENABLE_TRAINING_APIS=1"
        if "--enable_training_apis" in build_settings["build_params"]
        else "-DENABLE_TRAINING_APIS=0",
    ]

    # clean, build, and publish to a local directory
    subprocess.run([*gradle_command, "clean"], env=temp_env, shell=False, check=True, cwd=JAVA_ROOT)
    subprocess.run([*gradle_command, "build"], env=temp_env, shell=False, check=True, cwd=JAVA_ROOT)
    subprocess.run([*gradle_command, "publish"], env=temp_env, shell=False, check=True, cwd=JAVA_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Create Android Archive (AAR) package for one or more Android ABI(s)
        and building properties specified in the given build config file, see
        tools/ci_build/github/android/default_mobile_aar_build_settings.json for details.
        The output of the final AAR package can be found under [build_dir]/aar_out
        """,
    )

    parser.add_argument(
        "--android_sdk_path", type=str, default=os.environ.get("ANDROID_HOME", ""), help="Path to the Android SDK"
    )

    parser.add_argument(
        "--android_ndk_path", type=str, default=os.environ.get("ANDROID_NDK_HOME", ""), help="Path to the Android NDK"
    )

    parser.add_argument(
        "--build_dir",
        type=str,
        default=os.path.join(REPO_DIR, "build/android_aar"),
        help="Provide the root directory for build output",
    )

    parser.add_argument(
        "--include_ops_by_config",
        type=str,
        help="Include ops from config file. See /docs/Reduced_Operator_Kernel_build.md for more information.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="Release",
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration to build.",
    )

    parser.add_argument(
        "build_settings_file", type=pathlib.Path, help="Provide the file contains settings for building AAR"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Android SDK and NDK path are required
    if not args.android_sdk_path:
        raise ValueError("android_sdk_path is required")
    if not args.android_ndk_path:
        raise ValueError("android_ndk_path is required")

    _build_aar(args)


if __name__ == "__main__":
    main()
