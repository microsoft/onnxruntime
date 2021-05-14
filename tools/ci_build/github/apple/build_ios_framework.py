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

# We by default will build below 2 archs
DEFAULT_BUILD_OSX_ARCHS = [
    {'sysroot': 'iphoneos', 'arch': 'arm64'},
    {'sysroot': 'iphonesimulator', 'arch': 'x86_64'},
]


def _parse_build_settings(args):
    with open(args.build_settings_file.resolve()) as f:
        build_settings_data = json.load(f)

    build_settings = {}

    build_settings["build_osx_archs"] = build_settings_data.get("build_osx_archs", DEFAULT_BUILD_OSX_ARCHS)

    build_params = []
    if 'build_params' in build_settings_data:
        build_params += build_settings_data['build_params']
    else:
        raise ValueError('build_params is required in the build config file')

    build_settings['build_params'] = build_params
    return build_settings


def _build_package(args):
    build_settings = _parse_build_settings(args)
    build_dir = os.path.abspath(args.build_dir)

    # Temp dirs to hold building results
    intermediates_dir = os.path.join(build_dir, 'intermediates')
    build_config = args.config
    base_build_command = [sys.executable, BUILD_PY, '--config=' + build_config] + build_settings['build_params']

    # paths of the onnxruntime libraries for different archs
    ort_libs = []
    info_plist_path = ''

    # Build binary for each arch, one by one
    for osx_arch in build_settings['build_osx_archs']:
        sysroot = osx_arch['sysroot']
        current_arch = osx_arch['arch']
        build_dir_current_arch = os.path.join(intermediates_dir, sysroot + "_" + current_arch)
        build_command = base_build_command + [
            '--ios_sysroot=' + sysroot,
            '--osx_arch=' + current_arch,
            '--build_dir=' + build_dir_current_arch
        ]

        if args.include_ops_by_config is not None:
            build_command += ['--include_ops_by_config=' + str(args.include_ops_by_config.resolve())]

        # the actual build process for current arch
        subprocess.run(build_command, shell=False, check=True, cwd=REPO_DIR)

        # get the compiled lib path
        framework_dir = os.path.join(
            build_dir_current_arch, build_config, build_config + "-" + sysroot, 'onnxruntime.framework')
        ort_libs.append(os.path.join(framework_dir, 'onnxruntime'))

        # We actually only need to define the info.plist and headers once since they are all the same
        if not info_plist_path:
            info_plist_path = os.path.join(build_dir_current_arch, build_config, 'info.plist')
            headers = glob.glob(os.path.join(framework_dir, 'Headers', '*.h'))

    # manually create the fat framework
    framework_dir = os.path.join(build_dir, 'framework_out', 'onnxruntime.framework')
    pathlib.Path(framework_dir).mkdir(parents=True, exist_ok=True)

    # copy the header files and info.plist
    shutil.copy(info_plist_path, framework_dir)
    header_dir = os.path.join(framework_dir, 'Headers')
    pathlib.Path(header_dir).mkdir(parents=True, exist_ok=True)
    for _header in headers:
        shutil.copy(_header, header_dir)

    # use lipo to create a fat ort library
    lipo_command = ['lipo', '-create']
    lipo_command += ort_libs
    lipo_command += ['-output', os.path.join(framework_dir, 'onnxruntime')]
    subprocess.run(lipo_command, shell=False, check=True)


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

    parser.add_argument('--build_dir', type=pathlib.Path, default=os.path.join(REPO_DIR, 'build/iOS_framework'),
                        help='Provide the root directory for build output')

    parser.add_argument(
        "--include_ops_by_config", type=pathlib.Path,
        help="Include ops from config file. See /docs/Reduced_Operator_Kernel_build.md for more information.")

    parser.add_argument("--config", type=str, default="Release",
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration to build.")

    parser.add_argument('build_settings_file', type=pathlib.Path,
                        help='Provide the file contains settings for building iOS framework')

    args = parser.parse_args()

    if not args.build_settings_file.resolve().is_file():
        raise FileNotFoundError('Build config file {} is not a file.'.format(args.build_settings_file.resolve()))

    if args.include_ops_by_config is not None:
        include_ops_by_config_file = args.include_ops_by_config.resolve()
        if not include_ops_by_config_file.is_file():
            raise FileNotFoundError('Include ops config file {} is not a file.'.format(include_ops_by_config_file))

    return args


def main():
    args = parse_args()
    _build_package(args)


if __name__ == '__main__':
    main()
