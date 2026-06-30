#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
BUILD_PY = os.path.join(REPO_DIR, "tools", "ci_build", "build.py")

# We by default will build below 3 archs
DEFAULT_BUILD_OSX_ARCHS = {
    "iphoneos": ["arm64"],
    "iphonesimulator": ["arm64", "x86_64"],
}


def _parse_build_settings(args):
    with open(args.build_settings_file.resolve()) as f:
        build_settings_data = json.load(f)

    build_settings = {}

    build_settings["build_osx_archs"] = build_settings_data.get("build_osx_archs", DEFAULT_BUILD_OSX_ARCHS)

    if "build_params" in build_settings_data:
        build_settings["build_params"] = build_settings_data["build_params"]
    else:
        raise ValueError("build_params is required in the build config file")

    return build_settings


def _get_framework_dir(build_dir_current_arch, build_config, sysroot, build_dynamic_framework):
    if build_dynamic_framework:
        framework_subdir = "onnxruntime.framework"
    else:
        framework_subdir = os.path.join("static_framework", "onnxruntime.framework")
    candidates = []
    if sysroot == "macabi" and build_dynamic_framework:
        candidates.append(os.path.join(build_dir_current_arch, build_config, "onnxruntime.framework"))

    candidates.append(
        os.path.join(build_dir_current_arch, build_config, build_config + "-" + sysroot, framework_subdir)
    )

    for framework_dir in candidates:
        if os.path.exists(framework_dir):
            return framework_dir

    raise FileNotFoundError("Could not find built framework. Checked: " + ", ".join(candidates))


def _get_framework_binary_path(framework_dir):
    versioned_binary_path = os.path.join(framework_dir, "Versions", "A", "onnxruntime")
    if os.path.exists(versioned_binary_path):
        return versioned_binary_path

    return os.path.join(framework_dir, "onnxruntime")


def _get_framework_headers_path(framework_dir):
    versioned_headers_path = os.path.join(framework_dir, "Versions", "A", "Headers")
    if os.path.exists(versioned_headers_path):
        return versioned_headers_path

    return os.path.join(framework_dir, "Headers")


# Build fat framework for all archs of a single sysroot
# For example, arm64 and x86_64 for iphonesimulator
def _build_for_apple_sysroot(
    build_config, intermediates_dir, base_build_command, sysroot, archs, build_dynamic_framework
):
    # paths of the onnxruntime libraries for different archs
    ort_libs = []
    info_plist_path = ""

    # Build binary for each arch, one by one
    for current_arch in archs:
        build_dir_current_arch = os.path.join(intermediates_dir, sysroot + "_" + current_arch)
        # Use MacOS SDK for Catalyst builds
        apple_sysroot = "macosx" if sysroot == "macabi" else sysroot
        build_command = [
            *base_build_command,
            "--apple_sysroot=" + apple_sysroot,
            "--osx_arch=" + current_arch,
            "--build_dir=" + build_dir_current_arch,
        ]
        if current_arch == "x86_64" and sysroot in {"iphonesimulator", "macabi"}:
            build_command.extend(
                [
                    "--cmake_extra_defines=CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES=NO",
                    "--cmake_extra_defines=CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_EXPLICIT_MODULES=NO",
                ]
            )

        # the actual build process for current arch
        subprocess.run(build_command, shell=False, check=True, cwd=REPO_DIR)

        # get the compiled lib path
        framework_dir = _get_framework_dir(build_dir_current_arch, build_config, sysroot, build_dynamic_framework)
        ort_libs.append(_get_framework_binary_path(framework_dir))

        # We only need to copy Info.plist, framework_info.json, and headers once since they are the same
        if not info_plist_path:
            info_plist_path = os.path.join(build_dir_current_arch, build_config, "Info.plist")
            framework_info_path = os.path.join(build_dir_current_arch, build_config, "framework_info.json")
            headers = glob.glob(os.path.join(_get_framework_headers_path(framework_dir), "*.h"))

    bundle_root = os.path.join(intermediates_dir, "frameworks", sysroot)
    framework_dir = os.path.join(bundle_root, "onnxruntime.framework")
    static_library_dir = os.path.join(bundle_root, "onnxruntime")
    if build_dynamic_framework:
        if sysroot == "macosx" or sysroot == "macabi":
            fat_binary_output_path = os.path.join(framework_dir, "Versions", "A", "onnxruntime")
        else:
            fat_binary_output_path = os.path.join(framework_dir, "onnxruntime")
    else:
        fat_binary_output_path = os.path.join(static_library_dir, "libonnxruntime.a")

    if os.path.exists(bundle_root):
        shutil.rmtree(bundle_root)
    pathlib.Path(bundle_root).mkdir(parents=True, exist_ok=True)
    if build_dynamic_framework:
        pathlib.Path(framework_dir).mkdir(parents=True, exist_ok=True)

    # copy the Info.plist, framework_info.json, and header files

    # macos requires different framework structure:
    # https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPFrameworks/Concepts/FrameworkAnatomy.html
    if build_dynamic_framework and (sysroot == "macosx" or sysroot == "macabi"):
        # create headers and resources directory
        header_dir = os.path.join(framework_dir, "Versions", "A", "Headers")
        resource_dir = os.path.join(framework_dir, "Versions", "A", "Resources")
        pathlib.Path(header_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(resource_dir).mkdir(parents=True, exist_ok=True)

        shutil.copy(info_plist_path, resource_dir)
        shutil.copy(framework_info_path, os.path.dirname(framework_dir))

        for _header in headers:
            shutil.copy(_header, header_dir)

        # use lipo to create a fat ort library
        lipo_command = ["lipo", "-create"]
        lipo_command += ort_libs
        lipo_command += ["-output", fat_binary_output_path]
        subprocess.run(lipo_command, shell=False, check=True)

        # create the symbolic link
        pathlib.Path(os.path.join(framework_dir, "Versions", "Current")).symlink_to("A", target_is_directory=True)
        pathlib.Path(os.path.join(framework_dir, "Headers")).symlink_to(
            "Versions/Current/Headers", target_is_directory=True
        )
        pathlib.Path(os.path.join(framework_dir, "Resources")).symlink_to(
            "Versions/Current/Resources", target_is_directory=True
        )
        pathlib.Path(os.path.join(framework_dir, "onnxruntime")).symlink_to("Versions/Current/onnxruntime")

    else:
        if build_dynamic_framework:
            shutil.copy(info_plist_path, framework_dir)
            header_dir = os.path.join(framework_dir, "Headers")
        else:
            header_dir = os.path.join(static_library_dir, "Headers")
            pathlib.Path(static_library_dir).mkdir(parents=True, exist_ok=True)

        shutil.copy(framework_info_path, bundle_root)
        pathlib.Path(header_dir).mkdir(parents=True, exist_ok=True)

        for _header in headers:
            shutil.copy(_header, header_dir)

        # use lipo to create a fat ort library
        lipo_command = ["lipo", "-create"]
        lipo_command += ort_libs
        lipo_command += ["-output", fat_binary_output_path]
        subprocess.run(lipo_command, shell=False, check=True)

    return {
        "artifact_path": framework_dir if build_dynamic_framework else fat_binary_output_path,
        "framework_info_path": os.path.join(bundle_root, "framework_info.json"),
        "headers_path": header_dir if build_dynamic_framework else os.path.join(static_library_dir, "Headers"),
        "is_framework": build_dynamic_framework,
    }


def _merge_framework_info_files(files, output_file):
    merged_data = {}

    for file in files:
        with open(file) as f:
            data = json.load(f)
            for platform, values in data.items():
                assert platform not in merged_data, f"Duplicate platform value: {platform}"
                merged_data[platform] = values

    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)


def _build_package(args):
    build_settings = _parse_build_settings(args)
    build_dir = os.path.abspath(args.build_dir)

    # Temp dirs to hold building results
    intermediates_dir = os.path.join(build_dir, "intermediates")
    build_config = args.config

    # build framework for individual sysroot
    build_artifacts = []
    framework_info_files_to_merge = []
    public_headers_path = ""
    for sysroot in build_settings["build_osx_archs"]:
        base_build_command = (
            [sys.executable, BUILD_PY]
            + build_settings["build_params"]["base"]
            + build_settings["build_params"][sysroot]
            + ["--config=" + build_config]
        )

        if args.include_ops_by_config is not None:
            base_build_command += ["--include_ops_by_config=" + str(args.include_ops_by_config.resolve())]

        if args.path_to_protoc_exe is not None:
            base_build_command += ["--path_to_protoc_exe=" + str(args.path_to_protoc_exe.resolve())]

        build_artifact = _build_for_apple_sysroot(
            build_config,
            intermediates_dir,
            base_build_command,
            sysroot,
            build_settings["build_osx_archs"][sysroot],
            args.build_dynamic_framework,
        )
        build_artifacts.append(build_artifact)

        framework_info_files_to_merge.append(build_artifact["framework_info_path"])

        # headers for each sysroot are the same, pick one of them
        if not public_headers_path:
            public_headers_path = build_artifact["headers_path"]

    # create the folder for xcframework and copy the LICENSE and framework_info.json file
    xcframework_dir = os.path.join(build_dir, "framework_out")
    pathlib.Path(xcframework_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.join(REPO_DIR, "LICENSE"), xcframework_dir)
    shutil.copytree(public_headers_path, os.path.join(xcframework_dir, "Headers"), dirs_exist_ok=True, symlinks=True)
    _merge_framework_info_files(framework_info_files_to_merge, os.path.join(build_dir, "xcframework_info.json"))

    # remove existing xcframework if any
    xcframework_path = os.path.join(xcframework_dir, "onnxruntime.xcframework")
    if os.path.exists(xcframework_path):
        shutil.rmtree(xcframework_path)

    # Assemble the final xcframework
    build_xcframework_cmd = ["xcrun", "xcodebuild", "-create-xcframework", "-output", xcframework_path]
    for build_artifact in build_artifacts:
        if build_artifact["is_framework"]:
            build_xcframework_cmd.extend(["-framework", build_artifact["artifact_path"]])
        else:
            build_xcframework_cmd.extend(
                ["-library", build_artifact["artifact_path"], "-headers", build_artifact["headers_path"]]
            )

    subprocess.run(build_xcframework_cmd, shell=False, check=True, cwd=REPO_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Create iOS framework and podspec for one or more osx_archs (xcframework)
        and building properties specified in the given build config file, see
        tools/ci_build/github/apple/default_full_apple_framework_build_settings.json for details.
        The output of the final xcframework and podspec can be found under [build_dir]/framework_out.
        Please note, this building script will only work on macOS.
        """,
    )

    parser.add_argument(
        "--build_dir",
        type=pathlib.Path,
        default=os.path.join(REPO_DIR, "build/apple_framework"),
        help="Provide the root directory for build output",
    )

    parser.add_argument(
        "--include_ops_by_config",
        type=pathlib.Path,
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
        "--build_dynamic_framework",
        action="store_true",
        help="Build Dynamic Framework (default is build static framework).",
    )

    parser.add_argument(
        "build_settings_file", type=pathlib.Path, help="Provide the file contains settings for building iOS framework"
    )

    parser.add_argument("--path_to_protoc_exe", type=pathlib.Path, help="Path to protoc exe.")

    args = parser.parse_args()

    if not args.build_settings_file.resolve().is_file():
        raise FileNotFoundError(f"Build config file {args.build_settings_file.resolve()} is not a file.")

    if args.include_ops_by_config is not None:
        include_ops_by_config_file = args.include_ops_by_config.resolve()
        if not include_ops_by_config_file.is_file():
            raise FileNotFoundError(f"Include ops config file {include_ops_by_config_file} is not a file.")

    return args


def main():
    args = parse_args()
    _build_package(args)


if __name__ == "__main__":
    main()
