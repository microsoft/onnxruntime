#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pathlib
import json
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
BUILD_PY = os.path.join(REPO_DIR, "tools", "ci_build", "build.py")
JAVA_ROOT = os.path.join(REPO_DIR, "java")

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
    _setting_file = args.build_settings_file.resolve()

    if not _setting_file.is_file():
        raise FileNotFoundError('Build config file {} is not a file.'.format(_setting_file))

    with open(_setting_file) as f:
        _build_settings_data = json.load(f)

    build_settings = {}

    if 'build_abis' in _build_settings_data:
        build_settings['build_abis'] = _build_settings_data['build_abis']
    else:
        build_settings['build_abis'] = DEFAULT_BUILD_ABIS

    build_params = []
    if 'build_params' in _build_settings_data:
        build_params += _build_settings_data['build_params']
    else:
        raise ValueError('build_params is required in the build config file')

    if 'android_min_sdk_version' in _build_settings_data:
        build_settings['android_min_sdk_version'] = _build_settings_data['android_min_sdk_version']
    else:
        build_settings['android_min_sdk_version'] = DEFAULT_ANDROID_MIN_SDK_VER
    build_params += ['--android_api=' + str(build_settings['android_min_sdk_version'])]

    if 'android_target_sdk_version' in _build_settings_data:
        build_settings['android_target_sdk_version'] = _build_settings_data['android_target_sdk_version']
    else:
        build_settings['android_target_sdk_version'] = DEFAULT_ANDROID_TARGET_SDK_VER

    if build_settings['android_min_sdk_version'] > build_settings['android_target_sdk_version']:
        raise ValueError(
            'android_min_sdk_version {} cannot be larger than android_target_sdk_version {}'.format(
                build_settings['android_min_sdk_version'], build_settings['android_target_sdk_version']
            ))

    build_settings['build_params'] = build_params
    return build_settings


def _build_aar(args):
    build_settings = _parse_build_settings(args)
    build_dir = os.path.abspath(args.build_dir)

    # Setup temp environment for building
    _env = os.environ.copy()
    _env['ANDROID_HOME'] = os.path.abspath(args.android_sdk_path)
    _env['ANDROID_NDK_HOME'] = os.path.abspath(args.android_ndk_path)

    # Temp dirs to hold building results
    _intermediates_dir = os.path.join(build_dir, 'intermediates')
    _build_config = args.config
    _aar_dir = os.path.join(_intermediates_dir, 'aar', _build_config)
    _jnilibs_dir = os.path.join(_intermediates_dir, 'jnilibs', _build_config)
    _base_build_command = [
        sys.executable, BUILD_PY, '--config=' + _build_config
    ] + build_settings['build_params']

    # Build binary for each ABI, one by one
    for abi in build_settings['build_abis']:
        _build_dir = os.path.join(_intermediates_dir, abi)
        _build_command = _base_build_command + [
            '--android_abi=' + abi,
            '--build_dir=' + _build_dir
        ]

        if args.include_ops_by_config is not None:
            _build_command += ['--include_ops_by_config=' + args.include_ops_by_config]

        subprocess.run(_build_command, env=_env, shell=False, check=True, cwd=REPO_DIR)

        # create symbolic links for libonnxruntime.so and libonnxruntime4j_jni.so
        # to jnilibs/[abi] for later compiling the aar package
        _jnilibs_abi_dir = os.path.join(_jnilibs_dir, abi)
        os.makedirs(_jnilibs_abi_dir, exist_ok=True)
        for lib_name in ['libonnxruntime.so', 'libonnxruntime4j_jni.so']:
            _target_lib_name = os.path.join(_jnilibs_abi_dir, lib_name)
            # If the symbolic already exists, delete it first
            # For some reason, os.path.exists will return false for a symbolic link in Linux,
            # add double check with os.path.islink
            if os.path.exists(_target_lib_name) or os.path.islink(_target_lib_name):
                os.remove(_target_lib_name)
            os.symlink(os.path.join(_build_dir, _build_config, lib_name), _target_lib_name)

    # The directory to publish final AAR
    _aar_publish_dir = os.path.join(build_dir, 'aar_out', _build_config)
    os.makedirs(_aar_publish_dir, exist_ok=True)

    # get the common gradle command args
    _gradle_command = [
        'gradle',
        '--no-daemon',
        '-b=build-android.gradle',
        '-c=settings-android.gradle',
        '-DjniLibsDir=' + _jnilibs_dir,
        '-DbuildDir=' + _aar_dir,
        '-DpublishDir=' + _aar_publish_dir,
        '-DminSdkVer=' + str(build_settings['android_min_sdk_version']),
        '-DtargetSdkVer=' + str(build_settings['android_target_sdk_version'])
    ]

    # If not using shell on Window, will not be able to find gradle in path
    _shell = True if is_windows() else False

    # clean, build, and publish to a local directory
    subprocess.run(_gradle_command + ['clean'], env=_env, shell=_shell, check=True, cwd=JAVA_ROOT)
    subprocess.run(_gradle_command + ['build'], env=_env, shell=_shell, check=True, cwd=JAVA_ROOT)
    subprocess.run(_gradle_command + ['publish'], env=_env, shell=_shell, check=True, cwd=JAVA_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='''Create Android Archive (AAR) package for one or more Android ABI(s)
        and building properties specified in the given build config file, see
        tools/ci_build/github/android/default_mobile_aar_build_settings.json for details.
        The output of the final AAR package can be found under [build_dir]/aar_out
        '''
    )

    parser.add_argument("--android_sdk_path", type=str, default=os.environ.get("ANDROID_HOME", ""),
                        help="Path to the Android SDK")

    parser.add_argument("--android_ndk_path", type=str, default=os.environ.get("ANDROID_NDK_HOME", ""),
                        help="Path to the Android NDK")

    parser.add_argument('--build_dir', type=str, default=os.path.join(REPO_DIR, 'build/android_aar'),
                        help='Provide the root directory for build output')

    parser.add_argument(
        "--include_ops_by_config", type=str,
        help="Include ops from config file. See /docs/Reduced_Operator_Kernel_build.md for more information.")

    parser.add_argument("--config", type=str, default="Release",
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration to build.")

    parser.add_argument('build_settings_file', type=pathlib.Path,
                        help='Provide the file contains settings for building AAR')

    return parser.parse_args()


def main():
    args = parse_args()

    # Android SDK and NDK path are required
    if not args.android_sdk_path:
        raise ValueError('android_sdk_path is required')
    if not args.android_ndk_path:
        raise ValueError('android_ndk_path is required')

    _build_aar(args)


if __name__ == '__main__':
    main()
