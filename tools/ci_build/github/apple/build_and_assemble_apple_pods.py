#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import pathlib
import shutil
import sys
import tempfile

from c.assemble_c_pod_package import assemble_c_pod_package
from objectivec.assemble_objc_pod_package import assemble_objc_pod_package
from package_assembly_utils import PackageVariant, get_ort_version

SCRIPT_PATH = pathlib.Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_DIR = SCRIPT_PATH.parents[4]


logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger(SCRIPT_PATH.stem)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds an iOS framework and uses it to assemble iOS pod package files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=REPO_DIR / "build" / "apple_framework",
        help="The build directory. This will contain the iOS framework build output.",
    )
    parser.add_argument(
        "--staging-dir",
        type=pathlib.Path,
        default=REPO_DIR / "build" / "apple_pod_staging",
        help="The staging directory. This will contain the iOS pod package files. "
        "The pod package files do not have dependencies on files in the build directory.",
    )

    parser.add_argument(
        "--pod-version",
        default=f"{get_ort_version()}-local",
        help="The version string of the pod. The same version is used for all pods.",
    )

    parser.add_argument(
        "--variant",
        choices=PackageVariant.release_variant_names(),
        default=PackageVariant.Mobile.name,
        help="Pod package variant.",
    )

    parser.add_argument(
        "--platform-arch",
        nargs=2,
        action="append",
        metavar=("PLATFORM", "ARCH"),
        help="Specify a platform/arch pair to build. Repeat to specify multiple pairs. "
        "If no pairs are specified, all default supported pairs will be built.",
    )

    parser.add_argument("--test", action="store_true", help="Run tests on the framework and pod package files.")

    build_framework_group = parser.add_argument_group(
        title="iOS framework build arguments",
        description="See the corresponding arguments in build_apple_framework.py for details.",
    )

    build_framework_group.add_argument("--include-ops-by-config")
    build_framework_group.add_argument(
        "--build-settings-file", required=True, help="The positional argument of build_apple_framework.py."
    )
    build_framework_group.add_argument(
        "-b",
        "--build-apple-framework-arg",
        action="append",
        dest="build_apple_framework_extra_args",
        default=[],
        help="Pass an argument through to build_apple_framework.py. This may be specified multiple times.",
    )

    args = parser.parse_args()

    return args


def run(arg_list, cwd=None):
    import os
    import shlex
    import subprocess

    log.info(
        "Running subprocess in '{}'\n  {}".format(cwd or os.getcwd(), " ".join([shlex.quote(arg) for arg in arg_list]))
    )

    return subprocess.run(arg_list, check=True, cwd=cwd)


def main():
    args = parse_args()

    build_dir = args.build_dir.resolve()
    staging_dir = args.staging_dir.resolve()
    platform_arch = args.platform_arch.resolve()

    # build framework
    package_variant = PackageVariant[args.variant]
    framework_info_file = build_dir / "xcframework_info.json"

    log.info("Building Apple framework.")

    build_apple_framework_args = [
        sys.executable,
        str(SCRIPT_DIR / "build_apple_framework.py"),
        *args.build_apple_framework_extra_args,
    ]

    if args.include_ops_by_config is not None:
        build_apple_framework_args += ["--include_ops_by_config", args.include_ops_by_config]

    build_apple_framework_args += [
        "--build_dir",
        str(build_dir),
        "--platform_arch",
        platform_arch,
        args.build_settings_file,
    ]

    run(build_apple_framework_args)

    if args.test:
        test_apple_packages_args = [
            sys.executable,
            str(SCRIPT_DIR / "test_apple_packages.py"),
            "--fail_if_cocoapods_missing",
            "--framework_info_file",
            str(framework_info_file),
            "--c_framework_dir",
            str(build_dir / "framework_out"),
            "--variant",
            package_variant.name,
        ]

        run(test_apple_packages_args)

    # assemble pods and then move them to their target locations (staging_dir/<pod_name>)
    staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_dir) as pod_assembly_dir_name:
        pod_assembly_dir = pathlib.Path(pod_assembly_dir_name)

        log.info("Assembling C/C++ pod.")

        c_pod_staging_dir = pod_assembly_dir / "c_pod"
        c_pod_name, c_pod_podspec = assemble_c_pod_package(
            staging_dir=c_pod_staging_dir,
            pod_version=args.pod_version,
            framework_info_file=framework_info_file,
            framework_dir=build_dir / "framework_out" / "onnxruntime.xcframework",
            public_headers_dir=build_dir / "framework_out" / "Headers",
            package_variant=package_variant,
        )

        if args.test:
            test_c_pod_args = ["pod", "lib", "lint", "--verbose"]

            run(test_c_pod_args, cwd=c_pod_staging_dir)

        log.info("Assembling Objective-C pod.")

        objc_pod_staging_dir = pod_assembly_dir / "objc_pod"
        objc_pod_name, objc_pod_podspec = assemble_objc_pod_package(
            staging_dir=objc_pod_staging_dir,
            pod_version=args.pod_version,
            framework_info_file=framework_info_file,
            package_variant=package_variant,
        )

        if args.test:
            test_objc_pod_args = ["pod", "lib", "lint", "--verbose", f"--include-podspecs={c_pod_podspec}"]

            run(test_objc_pod_args, cwd=objc_pod_staging_dir)

        def move_dir(src, dst):
            if dst.is_dir():
                shutil.rmtree(dst)
            shutil.move(src, dst)

        move_dir(c_pod_staging_dir, staging_dir / c_pod_name)
        move_dir(objc_pod_staging_dir, staging_dir / objc_pod_name)

    log.info(f"Successfully assembled iOS pods at '{staging_dir}'.")


if __name__ == "__main__":
    main()
    main()
