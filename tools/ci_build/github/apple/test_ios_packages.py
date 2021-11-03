#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import contextlib
import os
import pathlib
import shutil
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))


from package_assembly_utils import (  # noqa: E402
    gen_file_from_template, load_framework_info)


def _test_ios_packages(args):
    # check if CocoaPods is installed
    if shutil.which('pod') is None:
        if args.fail_if_cocoapods_missing:
            raise ValueError('CocoaPods is required for this test')
        else:
            print('CocoaPods is not installed, ignore this test')
            return

    # Now we need to create a zip file contains the framework and the podspec file, both of these 2 files
    # should be under the c_framework_dir
    c_framework_dir = args.c_framework_dir.resolve()
    if not c_framework_dir.is_dir():
        raise FileNotFoundError('c_framework_dir {} is not a folder.'.format(c_framework_dir))

    has_framework = pathlib.Path(os.path.join(c_framework_dir, 'onnxruntime.framework')).exists()
    has_xcframework = pathlib.Path(os.path.join(c_framework_dir, 'onnxruntime.xcframework')).exists()

    if not has_framework and not has_xcframework:
        raise FileNotFoundError('{} does not have onnxruntime.framework/xcframework'.format(c_framework_dir))

    if has_framework and has_xcframework:
        raise ValueError('Cannot proceed when both onnxruntime.framework '
                         'and onnxruntime.xcframework exist')

    framework_name = 'onnxruntime.framework' if has_framework else 'onnxruntime.xcframework'

    # create a temp folder

    with contextlib.ExitStack() as context_stack:
        if args.test_project_stage_dir is None:
            stage_dir = context_stack.enter_context(tempfile.TemporaryDirectory())
        else:
            # If we specify the stage dir, then use it to create test project
            stage_dir = args.test_project_stage_dir
            if os.path.exists(stage_dir):
                shutil.rmtree(stage_dir)
            os.makedirs(stage_dir)

        # create a zip file contains the framework
        # TODO, move this into a util function
        local_pods_dir = os.path.join(stage_dir, 'local_pods')
        os.makedirs(local_pods_dir, exist_ok=True)
        # shutil.make_archive require target file as full path without extension
        zip_base_filename = os.path.join(local_pods_dir, 'onnxruntime-mobile-c')
        zip_file_path = zip_base_filename + '.zip'
        shutil.make_archive(zip_base_filename, 'zip', root_dir=c_framework_dir, base_dir=framework_name)

        # copy the test project to the temp_dir
        test_proj_path = os.path.join(REPO_DIR, 'onnxruntime', 'test', 'platform', 'ios', 'ios_package_test')
        target_proj_path = os.path.join(stage_dir, 'ios_package_test')
        shutil.copytree(test_proj_path, target_proj_path)

        # generate the podspec file from the template
        framework_info = load_framework_info(args.framework_info_file.resolve())

        with open(os.path.join(REPO_DIR, 'VERSION_NUMBER')) as version_file:
            ORT_VERSION = version_file.readline().strip()

        variable_substitutions = {
            "VERSION": ORT_VERSION,
            "IOS_DEPLOYMENT_TARGET": framework_info["IOS_DEPLOYMENT_TARGET"],
            "WEAK_FRAMEWORK": framework_info["WEAK_FRAMEWORK"],
            "LICENSE_FILE": '"LICENSE"',
        }

        podspec_template = os.path.join(SCRIPT_DIR, "c", "onnxruntime-mobile-c.podspec.template")
        podspec = os.path.join(target_proj_path, "onnxruntime-mobile-c.podspec")

        gen_file_from_template(podspec_template, podspec, variable_substitutions)

        # update the podspec to point to the local framework zip file
        with open(podspec, 'r') as file:
            file_data = file.read()
        file_data = file_data.replace('file:///http_source_placeholder', 'file:' + zip_file_path)

        # We will only publish xcframework, however, assembly of the xcframework is a post process
        # and it cannot be done by CMake for now. See, https://gitlab.kitware.com/cmake/cmake/-/issues/21752
        # For a single sysroot and arch built by build.py or cmake, we can only generate framework
        # We still need a way to test it, replace the xcframework with framework in the podspec
        if has_framework:
            file_data = file_data.replace('onnxruntime.xcframework', 'onnxruntime.framework')
        with open(podspec, 'w') as file:
            file.write(file_data)

        # clean the Cocoapods cache first, in case the same pod was cached in previous runs
        subprocess.run(['pod', 'cache', 'clean', '--all'], shell=False, check=True, cwd=target_proj_path)

        # install pods
        subprocess.run(['pod', 'install'], shell=False, check=True, cwd=target_proj_path)

        # run the tests
        if not args.prepare_test_project_only:
            subprocess.run(['xcrun', 'xcodebuild', 'test',
                            '-workspace', './ios_package_test.xcworkspace',
                            '-scheme', 'ios_package_test',
                            '-destination', 'platform=iOS Simulator,OS=latest,name=iPhone SE (2nd generation)'],
                           shell=False, check=True, cwd=target_proj_path)


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='Test iOS framework using CocoaPods package.'
    )

    parser.add_argument('--fail_if_cocoapods_missing', action='store_true',
                        help='This script will fail if CocoaPods is not installed, '
                        'will not throw error unless fail_if_cocoapod_missing is set.')

    parser.add_argument("--framework_info_file", type=pathlib.Path, required=True,
                        help="Path to the framework_info.json file containing additional values for the podspec. "
                             "This file should be generated by CMake in the build directory.")

    parser.add_argument('--c_framework_dir', type=pathlib.Path, required=True,
                        help='Provide the parent directory for C/C++ framework')

    parser.add_argument('--test_project_stage_dir', type=pathlib.Path,
                        help='The stage dir for the test project, if not specified, will use a temporary path')

    parser.add_argument('--prepare_test_project_only', action='store_true',
                        help='Prepare the test project only, without running the tests')

    return parser.parse_args()


def main():
    args = parse_args()
    _test_ios_packages(args)


if __name__ == '__main__':
    main()
