#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))


def _test_ios_packages(args):
    # check if CocoaPods is installed
    if shutil.which('pod') is None:
        if args.fail_if_cocoapods_missing:
            raise ValueError('CocoaPods is required for this test')
        else:
            print('CocoaPods is not installed, ignore this test')

    # Now we need to create a zip file contains the framework and the podspec file, both of these 2 files
    # should be under the c_framework_dir
    base_dir = args.c_framework_dir.resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError('c_framework_dir {} is not a folder.'.format(base_dir))

    podspec_path = os.path.join(base_dir, 'OnnxRuntimeBase.podspec')
    if not pathlib.Path(podspec_path).exists():
        raise FileNotFoundError('{} does not have OnnxRuntimeBase.podspec file'.format(base_dir))

    framework_path = os.path.join(base_dir, 'onnxruntime.framework')
    if not pathlib.Path(framework_path).exists():
        raise FileNotFoundError('{} does not have onnxruntime.framework'.format(base_dir))

    # create a temp folder under repo dir
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = '/Users/gwang/temp/eee'

        # create a zip file contains the framework and the podspec file
        zip_content_dir = os.path.join(temp_dir, 'zip_content')
        # zip_filename = os.path.join(temp_dir, 'OnnxRuntimeCAndCPP.zip')
        zip_base_filename = '/Users/gwang/temp/ddd'
        zip_filename = zip_base_filename + '.zip'
        os.makedirs(zip_content_dir, exist_ok=True)
        shutil.copy(podspec_path, zip_content_dir)
        framework_target_dir = os.path.join(zip_content_dir, 'onnxruntime.framework')
        if pathlib.Path(framework_target_dir).exists():
            shutil.rmtree(framework_target_dir)
        shutil.copytree(framework_path, framework_target_dir)
        shutil.make_archive(zip_base_filename, 'zip', zip_content_dir)

        # copy the test project to the temp_dir

        # Update the Podfile of the project and run pod install

        # Run the test

    print(args)


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='''Create iOS framework and podspec for one or more osx_archs (fat framework)
        and building properties specified in the given build config file, see
        tools/ci_build/github/apple/default_mobile_ios_framework_build_settings.json for details.
        The output of the final framework and podspec can be found under [build_dir]/framework_out.
        Please note, this building script will only work on macOS.
        '''
    )

    parser.add_argument('--fail_if_cocoapods_missing', action='store_true',
                        help='This script will fail if CocoaPods is not installed, '
                        'will not throw error unless fail_if_cocoapod_missing is set.')

    parser.add_argument('--c_framework_dir', type=pathlib.Path, required=True,
                        help='Provide the directory for C/C++ framework')

    return parser.parse_args()


def main():
    args = parse_args()
    _test_ios_packages(args)

if __name__ == '__main__':
    main()
