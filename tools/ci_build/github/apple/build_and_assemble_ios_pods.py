#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import pathlib
import sys


SCRIPT_PATH = pathlib.Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_DIR = SCRIPT_PATH.parents[4]

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s",
    level=logging.DEBUG)
log = logging.getLogger(SCRIPT_PATH.stem)


def ort_version():
    with open(REPO_DIR / "VERSION_NUMBER", mode="r") as version_file:
        return version_file.read().strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds an iOS framework and uses it to assemble iOS pod package files.")

    parser.add_argument("--build-dir", type=pathlib.Path, default=REPO_DIR / "build" / "ios_framework",
                        help="The build directory. This will contain the iOS framework build output.")
    parser.add_argument("--staging-dir", type=pathlib.Path, default=REPO_DIR / "build" / "ios_pod_staging",
                        help="The staging directory. This will contain the iOS pod package files. "
                             "The pod package files do not have dependencies on files in the build directory.")

    parser.add_argument("--pod-version", default=f"{ort_version()}-local",
                        help="The version string of the pod. The same version is used for all pods.")

    parser.add_argument("--test", action="store_true",
                        help="Run tests on the framework and pod package files.")

    build_framework_group = parser.add_argument_group(
        title="iOS framework build arguments",
        description="See the corresponding arguments in build_ios_framework.py for details.")

    build_framework_group.add_argument("--include-ops-by-config")
    build_framework_group.add_argument("--build-settings-file", required=True,
                                       help="The positional argument of build_ios_framework.py.")
    build_framework_group.add_argument("-b", "--build-ios-framework-arg", action="append",
                                       dest="build_ios_framework_extra_args", default=[],
                                       help="Pass an argument through to build_ios_framework.py. "
                                            "This may be specified multiple times.")

    args = parser.parse_args()

    return args


def run(arg_list, cwd=None):
    import os
    import shlex
    import subprocess

    log.info("Running subprocess in '{0}'\n  {1}".format(
        cwd or os.getcwd(), " ".join([shlex.quote(arg) for arg in arg_list])))

    return subprocess.run(arg_list, check=True, cwd=cwd)


def main():
    args = parse_args()

    build_dir = args.build_dir.resolve()
    staging_dir = args.staging_dir.resolve()

    log.info("Building iOS framework.")

    build_ios_framework_args = \
        [sys.executable, str(SCRIPT_DIR / "build_ios_framework.py")] + args.build_ios_framework_extra_args

    if args.include_ops_by_config is not None:
        build_ios_framework_args += ["--include_ops_by_config", args.include_ops_by_config]

    build_ios_framework_args += ["--build_dir", str(build_dir),
                                 args.build_settings_file]

    run(build_ios_framework_args)

    if args.test:
        test_ios_packages_args = [sys.executable, str(SCRIPT_DIR / "test_ios_packages.py"),
                                  "--fail_if_cocoapods_missing",
                                  "--framework_info_file", str(build_dir / "framework_info.json"),
                                  "--c_framework_dir", str(build_dir / "framework_out")]

        run(test_ios_packages_args)

    log.info("Assembling onnxruntime-mobile-c pod.")

    assemble_c_pod_args = [sys.executable, str(SCRIPT_DIR / "c" / "assemble_c_pod_package.py"),
                           "--staging-dir", str(staging_dir / "onnxruntime-mobile-c"),
                           "--pod-version", args.pod_version,
                           "--framework-info-file", str(build_dir / "framework_info.json"),
                           "--framework-dir", str(build_dir / "framework_out" / "onnxruntime.xcframework"),
                           "--public-headers-dir", str(build_dir / "framework_out" / "Headers")]

    run(assemble_c_pod_args)

    if args.test:
        test_c_pod_args = ["pod", "lib", "lint", "--verbose"]

        run(test_c_pod_args, cwd=staging_dir / "onnxruntime-mobile-c")

    log.info("Assembling onnxruntime-mobile-objc pod.")

    assemble_objc_pod_args = [sys.executable, str(SCRIPT_DIR / "objectivec" / "assemble_objc_pod_package.py"),
                              "--staging-dir", str(staging_dir / "onnxruntime-mobile-objc"),
                              "--pod-version", args.pod_version,
                              "--framework-info-file", str(build_dir / "framework_info.json")]

    run(assemble_objc_pod_args)

    if args.test:
        c_podspec_file = staging_dir / "onnxruntime-mobile-c" / "onnxruntime-mobile-c.podspec"
        test_objc_pod_args = ["pod", "lib", "lint", "--verbose", f"--include-podspecs={c_podspec_file}"]

        run(test_objc_pod_args, cwd=staging_dir / "onnxruntime-mobile-objc")

    log.info(f"Successfully assembled iOS pods at '{staging_dir}'.")


if __name__ == "__main__":
    main()
