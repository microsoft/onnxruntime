#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass

from build_settings_utils import get_build_params, get_sysroot_arch_pairs, parse_build_settings_file

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
REPO_DIR = SCRIPT_DIR.parents[3]
BUILD_PY = REPO_DIR / "tools" / "ci_build" / "build.py"


def _filter_sysroot_arch_pairs(
    all_sysroot_arch_pairs: list[tuple[str, str]],
    args,
) -> list[tuple[str, str]]:
    if args.only_build_single_sysroot_arch_framework is not None:
        specified_sysroot_arch_pair = (
            args.only_build_single_sysroot_arch_framework[0],
            args.only_build_single_sysroot_arch_framework[1],
        )
        if specified_sysroot_arch_pair not in all_sysroot_arch_pairs:
            raise ValueError(
                "Sysroot/arch pair is not present in build settings file. "
                f"Specified: {specified_sysroot_arch_pair}, available: {all_sysroot_arch_pairs}"
            )

        return [specified_sysroot_arch_pair]

    return all_sysroot_arch_pairs.copy()


# info related to a framework for a single sysroot and arch (e.g., iphoneos/arm64)
@dataclass
class SysrootArchFrameworkInfo:
    framework_dir: pathlib.Path
    framework_info_file: pathlib.Path
    info_plist_file: pathlib.Path


# find or build the sysroot/arch framework
# if `base_build_command` is not None, the framework will be built
def _find_or_build_sysroot_arch_framework(
    build_config: str,
    intermediates_dir: pathlib.Path,
    base_build_command: list[str] | None,
    sysroot: str,
    arch: str,
    build_dynamic_framework: bool,
) -> SysrootArchFrameworkInfo:
    do_build = base_build_command is not None

    build_dir_current_arch = intermediates_dir / f"{sysroot}_{arch}"

    if do_build:
        # Use MacOS SDK for Catalyst builds
        apple_sysroot = "macosx" if sysroot == "macabi" else sysroot
        build_command = [
            *base_build_command,
            f"--apple_sysroot={apple_sysroot}",
            f"--osx_arch={arch}",
            f"--build_dir={build_dir_current_arch}",
        ]

        # build framework for specified sysroot/arch
        subprocess.run(build_command, shell=False, check=True)

    # get the compiled lib path
    framework_dir = pathlib.Path.joinpath(
        build_dir_current_arch,
        build_config,
        build_config + "-" + sysroot,
        (
            "onnxruntime.framework"
            if build_dynamic_framework
            else pathlib.Path("static_framework/onnxruntime.framework")
        ),
    )

    info_plist_file = build_dir_current_arch / build_config / "Info.plist"
    framework_info_file = build_dir_current_arch / build_config / "framework_info.json"

    if not do_build:
        for expected_path in [framework_dir, info_plist_file, framework_info_file]:
            if not expected_path.exists():
                raise FileNotFoundError(f"Expected framework path does not exist: {expected_path}")

    return SysrootArchFrameworkInfo(
        framework_dir=framework_dir,
        info_plist_file=info_plist_file,
        framework_info_file=framework_info_file,
    )


def _write_sysroot_arch_framework_build_outputs_to_file(
    build_dir: pathlib.Path,
    built_sysroot_arch_framework_infos: list[SysrootArchFrameworkInfo],
    build_outputs_file_path: pathlib.Path,
):
    with open(build_outputs_file_path, mode="w") as build_outputs_file:

        def write_path(p: pathlib.Path):
            print(p.resolve().relative_to(build_dir), file=build_outputs_file)

        for info in built_sysroot_arch_framework_infos:
            write_path(info.framework_dir)
            write_path(info.framework_info_file)
            write_path(info.info_plist_file)


# info related to a fat framework for a single sysroot (e.g., iphoneos)
# a fat framework contains one or more frameworks for individual archs (e.g., arm64)
@dataclass
class SysrootFrameworkInfo:
    framework_dir: pathlib.Path
    framework_info_file: pathlib.Path


# Assemble fat framework for all archs of a single sysroot
# For example, arm64 and x86_64 for iphonesimulator
def _assemble_fat_framework_for_sysroot(
    intermediates_dir: pathlib.Path, sysroot: str, sysroot_arch_framework_infos: list[SysrootArchFrameworkInfo]
) -> SysrootFrameworkInfo:
    assert len(sysroot_arch_framework_infos) > 0, "There must be at least one sysroot arch framework."

    # paths of the onnxruntime libraries for different archs
    ort_libs = [(info.framework_dir / "onnxruntime") for info in sysroot_arch_framework_infos]

    # We only need to copy Info.plist, framework_info.json, and headers once since they are the same
    info_plist_path = sysroot_arch_framework_infos[0].info_plist_file
    framework_info_path = sysroot_arch_framework_infos[0].framework_info_file
    header_paths = (sysroot_arch_framework_infos[0].framework_dir / "Headers").glob("*.h")

    # manually create the fat framework
    framework_dir = intermediates_dir / "frameworks" / sysroot / "onnxruntime.framework"
    # remove the existing framework if any
    if framework_dir.exists():
        shutil.rmtree(framework_dir)
    framework_dir.mkdir(parents=True, exist_ok=True)

    # macos requires different framework structure:
    # https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPFrameworks/Concepts/FrameworkAnatomy.html
    use_macos_framework_structure = sysroot == "macosx" or sysroot == "macabi"

    if use_macos_framework_structure:
        dst_header_dir = framework_dir / "Versions" / "A" / "Headers"
        dst_resource_dir = framework_dir / "Versions" / "A" / "Resources"
        dst_header_dir.mkdir(parents=True, exist_ok=True)
        dst_resource_dir.mkdir(parents=True, exist_ok=True)

        dst_info_plist_path = dst_resource_dir / info_plist_path.name
        dst_framework_info_path = framework_dir.parent / framework_info_path.name

        dst_fat_library_path = framework_dir / "Versions" / "A" / "onnxruntime"
    else:
        dst_header_dir = framework_dir / "Headers"
        dst_header_dir.mkdir(parents=True, exist_ok=True)

        dst_info_plist_path = framework_dir / info_plist_path.name
        dst_framework_info_path = framework_dir.parent / framework_info_path.name

        dst_fat_library_path = framework_dir / "onnxruntime"

    # copy the Info.plist, framework_info.json, and header files
    shutil.copy(info_plist_path, dst_info_plist_path)
    shutil.copy(framework_info_path, dst_framework_info_path)

    for header_path in header_paths:
        shutil.copy(header_path, dst_header_dir)

    # use lipo to create a fat ort library
    lipo_command = ["lipo", "-create"]
    lipo_command += [str(lib) for lib in ort_libs]
    lipo_command += ["-output", str(dst_fat_library_path)]
    subprocess.run(lipo_command, shell=False, check=True)

    if use_macos_framework_structure:
        # create additional symbolic links
        (framework_dir / "Versions" / "Current").symlink_to("A", target_is_directory=True)
        (framework_dir / "Headers").symlink_to("Versions/Current/Headers", target_is_directory=True)
        (framework_dir / "Resources").symlink_to("Versions/Current/Resources", target_is_directory=True)
        (framework_dir / "onnxruntime").symlink_to("Versions/Current/onnxruntime")

    return SysrootFrameworkInfo(
        framework_dir=framework_dir,
        framework_info_file=dst_framework_info_path,
    )


def _merge_framework_info_files(files: list[pathlib.Path], output_file: pathlib.Path):
    merged_data = {}

    for file in files:
        with open(file) as f:
            data = json.load(f)
            for platform, values in data.items():
                assert platform not in merged_data, f"Duplicate platform value: {platform}"
                merged_data[platform] = values

    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)


def _assemble_xcframework(build_dir: pathlib.Path, sysroot_framework_infos: list[SysrootFrameworkInfo]) -> pathlib.Path:
    assert len(sysroot_framework_infos) > 0, "There must be at least one sysroot fat framework."

    framework_dirs = [info.framework_dir for info in sysroot_framework_infos]
    framework_info_files_to_merge = [info.framework_info_file for info in sysroot_framework_infos]

    # headers for each sysroot are the same, pick the first one
    public_headers_path = sysroot_framework_infos[0].framework_dir / "Headers"

    # create the output folder for the xcframework and copy the LICENSE and header files and generate the
    # xcframework_info.json file
    output_dir = build_dir / "framework_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_DIR / "LICENSE", output_dir)
    shutil.copytree(public_headers_path, output_dir / "Headers", dirs_exist_ok=True, symlinks=True)
    _merge_framework_info_files(framework_info_files_to_merge, build_dir / "xcframework_info.json")

    # remove existing xcframework if any
    xcframework_path = output_dir / "onnxruntime.xcframework"
    if os.path.exists(xcframework_path):
        shutil.rmtree(xcframework_path)

    # Assemble the final xcframework
    build_xcframework_cmd = ["xcrun", "xcodebuild", "-create-xcframework", "-output", str(xcframework_path)]
    for framework_dir in framework_dirs:
        build_xcframework_cmd.extend(["-framework", str(framework_dir)])

    subprocess.run(build_xcframework_cmd, shell=False, check=True, cwd=REPO_DIR)

    return xcframework_path


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
        default=(REPO_DIR / "build" / "apple_framework"),
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

    parser.add_argument("--path_to_protoc_exe", type=pathlib.Path, help="Path to protoc exe.")

    parser.add_argument(
        "build_settings_file", type=pathlib.Path, help="Provide the file contains settings for building iOS framework"
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--only_build_single_sysroot_arch_framework",
        nargs=2,
        metavar=("sysroot", "arch"),
        help="Only build the specified sysroot/arch framework. E.g., sysroot = iphoneos, arch = arm64. "
        "This can be used to split up the builds between different invocations of this script. "
        "The sysroot and arch combination should be one from the build settings file.",
    )

    mode_group.add_argument(
        "--only_assemble_xcframework",
        action="store_true",
        help="Only assemble the xcframework (and intermediate fat frameworks) from the sysroot/arch frameworks. "
        "This mode requires the necessary sysroot/arch frameworks to already be present in the build directory, "
        "as if this script were previously invoked with `--only_build_single_sysroot_arch_framework` and the same "
        "`--build_dir` value.",
    )

    parser.add_argument(
        "--record_sysroot_arch_framework_build_outputs_to_file",
        type=pathlib.Path,
        help="If building sysroot/arch framework(s), write the build output file paths to the specified file. "
        "The paths will be relative to the build directory specified by `--build_dir`. "
        "These build output files are the files that should be preserved between split-build invocations with "
        "`--only_build_single_sysroot_arch_framework` and `--only_assemble_xcframework`.",
    )

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

    build_settings_file = args.build_settings_file.resolve()
    build_settings = parse_build_settings_file(build_settings_file)

    build_dir = args.build_dir.resolve()
    build_config = args.config

    all_sysroot_arch_pairs = get_sysroot_arch_pairs(build_settings)

    # default to building frameworks and assembling xcframework
    do_sysroot_arch_framework_build = do_xcframework_assembly = True

    # mode options may modify the default behavior
    if args.only_build_single_sysroot_arch_framework is not None:
        do_xcframework_assembly = False
    if args.only_assemble_xcframework:
        do_sysroot_arch_framework_build = False

    # directory for intermediate build files
    intermediates_dir = build_dir / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    sysroot_to_sysroot_arch_framework_infos: dict[str, list[SysrootArchFrameworkInfo]] = {}

    if do_sysroot_arch_framework_build:
        # build sysroot/arch frameworks
        sysroot_arch_pairs_to_build = _filter_sysroot_arch_pairs(all_sysroot_arch_pairs, args)

        # common build command trailing args
        build_command_trailing_args = [f"--config={build_config}"]
        if args.include_ops_by_config is not None:
            build_command_trailing_args += [f"--include_ops_by_config={args.include_ops_by_config.resolve()}"]
        if args.path_to_protoc_exe is not None:
            build_command_trailing_args += [f"--path_to_protoc_exe={args.path_to_protoc_exe.resolve()}"]

        built_sysroot_arch_framework_infos: list[SysrootArchFrameworkInfo] = []

        for sysroot, arch in sysroot_arch_pairs_to_build:
            infos_for_sysroot = sysroot_to_sysroot_arch_framework_infos.setdefault(sysroot, [])
            base_build_command = [
                sys.executable,
                BUILD_PY,
                *get_build_params(build_settings, "base"),
                *get_build_params(build_settings, sysroot),
                *build_command_trailing_args,
            ]

            info = _find_or_build_sysroot_arch_framework(
                build_config, intermediates_dir, base_build_command, sysroot, arch, args.build_dynamic_framework
            )
            infos_for_sysroot.append(info)
            built_sysroot_arch_framework_infos.append(info)

            print(f"Built sysroot/arch framework for {sysroot}/{arch}: {info.framework_dir}")

        if args.record_sysroot_arch_framework_build_outputs_to_file is not None:
            _write_sysroot_arch_framework_build_outputs_to_file(
                build_dir, built_sysroot_arch_framework_infos, args.record_sysroot_arch_framework_build_outputs_to_file
            )

    else:
        # do not build sysroot/arch frameworks, but look for existing ones
        for sysroot, arch in all_sysroot_arch_pairs:
            infos_for_sysroot = sysroot_to_sysroot_arch_framework_infos.setdefault(sysroot, [])
            base_build_command = None  # do not build anything
            info = _find_or_build_sysroot_arch_framework(
                build_config, intermediates_dir, base_build_command, sysroot, arch, args.build_dynamic_framework
            )
            infos_for_sysroot.append(info)

            print(f"Found existing sysroot/arch framework for {sysroot}/{arch}: {info.framework_dir}")

    if do_xcframework_assembly:
        sysroot_framework_infos: list[SysrootFrameworkInfo] = []

        # assemble fat frameworks
        for sysroot, sysroot_arch_framework_infos in sysroot_to_sysroot_arch_framework_infos.items():
            sysroot_framework_info = _assemble_fat_framework_for_sysroot(
                intermediates_dir, sysroot, sysroot_arch_framework_infos
            )
            sysroot_framework_infos.append(sysroot_framework_info)

        # assemble xcframework
        xcframework_dir = _assemble_xcframework(build_dir, sysroot_framework_infos)

        print(f"Assembled xcframework: {xcframework_dir}")


if __name__ == "__main__":
    main()
