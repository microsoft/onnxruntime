#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from c.assemble_c_pod_package import assemble_c_pod_package
from package_assembly_utils import PackageVariant, gen_file_from_template, get_ort_version

SCRIPT_PATH = pathlib.Path(__file__).resolve(strict=True)
REPO_DIR = SCRIPT_PATH.parents[4]


def _test_apple_packages(args):
    # check if CocoaPods is installed
    if shutil.which("pod") is None:
        if args.fail_if_cocoapods_missing:
            raise ValueError("CocoaPods is required for this test")
        else:
            print("CocoaPods is not installed, ignore this test")
            return

    # Now we need to create a zip file contains the framework and the podspec file, both of these 2 files
    # should be under the c_framework_dir
    c_framework_dir = args.c_framework_dir.resolve()
    if not c_framework_dir.is_dir():
        raise FileNotFoundError(f"c_framework_dir {c_framework_dir} is not a folder.")

    has_framework = (c_framework_dir / "onnxruntime.framework").exists()
    has_xcframework = (c_framework_dir / "onnxruntime.xcframework").exists()

    if not has_framework and not has_xcframework:
        raise FileNotFoundError(f"{c_framework_dir} does not have onnxruntime.framework/xcframework")

    if has_framework and has_xcframework:
        raise ValueError("Cannot proceed when both onnxruntime.framework and onnxruntime.xcframework exist")

    framework_name = "onnxruntime.framework" if has_framework else "onnxruntime.xcframework"

    # create a temp folder

    with contextlib.ExitStack() as context_stack:
        if args.test_project_stage_dir is None:
            stage_dir = pathlib.Path(context_stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        else:
            # If we specify the stage dir, then use it to create test project
            stage_dir = args.test_project_stage_dir.resolve()
            if os.path.exists(stage_dir):
                shutil.rmtree(stage_dir)
            os.makedirs(stage_dir)

        # assemble the test project here
        target_proj_path = stage_dir / "apple_package_test"

        # copy the test project source files to target_proj_path
        test_proj_path = pathlib.Path(REPO_DIR, "onnxruntime/test/platform/apple/apple_package_test")
        shutil.copytree(test_proj_path, target_proj_path)

        # assemble local pod files here
        local_pods_dir = stage_dir / "local_pods"

        # We will only publish xcframework, however, assembly of the xcframework is a post process
        # and it cannot be done by CMake for now. See, https://gitlab.kitware.com/cmake/cmake/-/issues/21752
        # For a single sysroot and arch built by build.py or cmake, we can only generate framework
        # We still need a way to test it. framework_dir and public_headers_dir have different values when testing a
        # framework and a xcframework.
        framework_dir = args.c_framework_dir / framework_name
        public_headers_dir = framework_dir / "Headers" if has_framework else args.c_framework_dir / "Headers"

        pod_name, podspec = assemble_c_pod_package(
            staging_dir=local_pods_dir,
            pod_version=get_ort_version(),
            framework_info_file=args.framework_info_file,
            public_headers_dir=public_headers_dir,
            framework_dir=framework_dir,
            package_variant=PackageVariant[args.variant],
        )

        # move podspec out to target_proj_path first
        podspec = shutil.move(podspec, target_proj_path / podspec.name)

        # create a zip file contains the framework
        zip_file_path = local_pods_dir / f"{pod_name}.zip"
        # shutil.make_archive require target file as full path without extension
        shutil.make_archive(zip_file_path.with_suffix(""), "zip", root_dir=local_pods_dir)

        # update the podspec to point to the local framework zip file
        with open(podspec) as file:
            file_data = file.read()

        file_data = file_data.replace("file:///http_source_placeholder", f"file:///{zip_file_path}")

        with open(podspec, "w") as file:
            file.write(file_data)

        # generate Podfile to point to pod
        gen_file_from_template(
            target_proj_path / "Podfile.template",
            target_proj_path / "Podfile",
            {"C_POD_NAME": pod_name, "C_POD_PODSPEC": f"./{podspec.name}"},
        )

        # clean the Cocoapods cache first, in case the same pod was cached in previous runs
        subprocess.run(["pod", "cache", "clean", "--all"], shell=False, check=True, cwd=target_proj_path)

        # install pods
        subprocess.run(["pod", "install"], shell=False, check=True, cwd=target_proj_path)

        # run the tests
        if not args.prepare_test_project_only:
            simulator_device_info = subprocess.check_output(
                [
                    sys.executable,
                    str(REPO_DIR / "tools" / "ci_build" / "github" / "apple" / "get_simulator_device_info.py"),
                ],
                text=True,
            ).strip()
            print(f"Simulator device info:\n{simulator_device_info}")

            simulator_device_info = json.loads(simulator_device_info)

            subprocess.run(
                [
                    "xcrun",
                    "xcodebuild",
                    "test",
                    "-workspace",
                    "./apple_package_test.xcworkspace",
                    "-scheme",
                    "ios_package_test",
                    "-destination",
                    f"platform=iOS Simulator,id={simulator_device_info['device_udid']}",
                ],
                shell=False,
                check=True,
                cwd=target_proj_path,
            )

            if PackageVariant[args.variant] != PackageVariant.Mobile:
                subprocess.run(
                    [
                        "xcrun",
                        "xcodebuild",
                        "test",
                        "-workspace",
                        "./apple_package_test.xcworkspace",
                        "-scheme",
                        "macos_package_test",
                        "-destination",
                        "platform=macos",
                    ],
                    shell=False,
                    check=True,
                    cwd=target_proj_path,
                )


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Test iOS framework using CocoaPods package."
    )

    parser.add_argument(
        "--fail_if_cocoapods_missing",
        action="store_true",
        help="This script will fail if CocoaPods is not installed, "
        "will not throw error unless fail_if_cocoapod_missing is set.",
    )

    parser.add_argument(
        "--framework_info_file",
        type=pathlib.Path,
        required=True,
        help="Path to the framework_info.json or xcframework_info.json file containing additional values for the podspec. "
        "This file should be generated by CMake in the build directory.",
    )

    parser.add_argument(
        "--c_framework_dir", type=pathlib.Path, required=True, help="Provide the parent directory for C/C++ framework"
    )

    parser.add_argument(
        "--variant",
        choices=PackageVariant.all_variant_names(),
        default=PackageVariant.Test.name,
        help="Pod package variant.",
    )

    parser.add_argument(
        "--test_project_stage_dir",
        type=pathlib.Path,
        help="The stage dir for the test project, if not specified, will use a temporary path",
    )

    parser.add_argument(
        "--prepare_test_project_only",
        action="store_true",
        help="Prepare the test project only, without running the tests",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    _test_apple_packages(args)


if __name__ == "__main__":
    main()
