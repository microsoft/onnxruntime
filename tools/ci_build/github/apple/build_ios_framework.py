#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pathlib
import json
import subprocess
import sys
import glob
import shutil

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
BUILD_PY = os.path.join(REPO_DIR, "tools", "ci_build", "build.py")

# We by default will build below 2 archs
DEFAULT_BUILD_OSX_ARCHS = [["iphonesimulator", "x86_64"], ["iphoneos", "arm64"]]


def _parse_build_settings(args):
    _setting_file = args.build_settings_file.resolve()

    if not _setting_file.is_file():
        raise FileNotFoundError('Build config file {} is not a file.'.format(_setting_file))

    with open(_setting_file) as f:
        _build_settings_data = json.load(f)

    build_settings = {}

    if 'build_osx_archs' in _build_settings_data:
        build_settings['build_osx_archs'] = []
        for _arch in _build_settings_data['build_osx_archs']:
            build_settings['build_osx_archs'].append([_arch["sysroot"], _arch["arch"]])
    else:
        raise ValueError('build_osx_archs is required in the build config file')

    build_params = []
    if 'build_params' in _build_settings_data:
        build_params += _build_settings_data['build_params']
    else:
        raise ValueError('build_params is required in the build config file')

    build_settings['build_params'] = build_params
    return build_settings


def _build_package(args):
    build_settings = _parse_build_settings(args)
    build_dir = os.path.abspath(args.build_dir)

    # Temp dirs to hold building results
    _intermediates_dir = os.path.join(build_dir, 'intermediates')
    _build_config = args.config
    _base_build_command = [
        sys.executable, BUILD_PY,
        '--config=' + _build_config,
        '--apple_deploy_target=' + args.apple_deploy_target
    ] + build_settings['build_params']

    # paths of the onnxruntime libraries for different archs
    _ort_libs = []
    _info_plist = ''

    # Build binary for each arch, one by one
    for _osx_arch in build_settings['build_osx_archs']:
        _build_dir = os.path.join(_intermediates_dir, _osx_arch[0] + "_" + _osx_arch[1])
        _build_command = _base_build_command + [
            '--ios_sysroot=' + _osx_arch[0],
            '--osx_arch=' + _osx_arch[1],
            '--build_dir=' + _build_dir
        ]

        if args.include_ops_by_config is not None:
            _build_command += ['--include_ops_by_config=' + args.include_ops_by_config]

        subprocess.run(_build_command, shell=False, check=True, cwd=REPO_DIR)

        _framework_dir = os.path.join(
            _build_dir, _build_config, _build_config + "-" + _osx_arch[0], 'onnxruntime.framework')
        _ort_libs.append(os.path.join(_framework_dir, 'onnxruntime'))

        # We actually only need to define the info.plist and headers once since them are all the same
        if not _info_plist:
            _info_plist = os.path.join(_build_dir, _build_config, 'info.plist')
            _headers = glob.glob(os.path.join(_framework_dir, 'Headers', '*.h'))

    # manually create the fat framework
    _framework_dir = os.path.join(build_dir, 'framework_out', 'onnxruntime.framework')
    pathlib.Path(_framework_dir).mkdir(parents=True, exist_ok=True)

    # copy the header files and info.plist
    shutil.copy(_info_plist, _framework_dir)
    _header_dir = os.path.join(_framework_dir, 'Headers')
    pathlib.Path(_header_dir).mkdir(parents=True, exist_ok=True)
    for _header in _headers:
        shutil.copy(_header, _header_dir)

    # use lipo to create a fat ort library
    _lipo_command = ['lipo', '-create']
    _lipo_command += _ort_libs
    _lipo_command += ['-output', os.path.join(_framework_dir, 'onnxruntime')]
    subprocess.run(_lipo_command, shell=False, check=True)


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

    parser.add_argument("--apple_deploy_target", type=str,
                        help="Specify the minimum version of the target iOS platform")

    parser.add_argument('--build_dir', type=str, default=os.path.join(REPO_DIR, 'build/iOS_framework'),
                        help='Provide the root directory for build output')

    parser.add_argument(
        "--include_ops_by_config", type=str,
        help="Include ops from config file. See /docs/Reduced_Operator_Kernel_build.md for more information.")

    parser.add_argument("--config", type=str, default="Release",
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration to build.")

    parser.add_argument('build_settings_file', type=pathlib.Path,
                        help='Provide the file contains settings for building iOS framework')

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.apple_deploy_target:
        raise ValueError('apple_deploy_target is required')

    _build_package(args)


if __name__ == '__main__':
    main()
