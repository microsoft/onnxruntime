#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

import os
import sys
import shutil
import subprocess
import hashlib
from os.path import expanduser


test_data_url = 'https://onnxruntimetestdata.blob.core.windows.net/models/cmake-3.13.2-win64-x64.zip'
test_data_checksum = '4cbaf72047d20bc84742327a5eafffd1'


def check_md5(filename, expected_md5):
    if not os.path.exists(filename):
        return False
    hash_md5 = hashlib.md5()
    BLOCKSIZE = 1024*64
    with open(filename, "rb") as f:
        buf = f.read(BLOCKSIZE)
        while len(buf) > 0:
            hash_md5.update(buf)
            buf = f.read(BLOCKSIZE)
    hex = hash_md5.hexdigest()
    if hex != expected_md5:
        print('md5 mismatch, expect %s, got %s' % (expected_md5, hex))
        os.remove(filename)
        return False
    return True


def is_windows():
    return sys.platform.startswith("win")

# the last part of src_url should be unique, across all the builds


def download_test_data(models_dir, src_url, expected_md5):
    cache_dir = os.path.join(expanduser("~"), '.cache', 'onnxruntime')
    os.makedirs(cache_dir, exist_ok=True)
    local_zip_file = os.path.join(cache_dir, os.path.basename(src_url))
    if not check_md5(local_zip_file, expected_md5):
        print("Downloading test data")
        if is_windows():
            subprocess.run(['powershell', '-Command', 'Invoke-WebRequest %s -OutFile %s' % (src_url, local_zip_file)],
                           check=True)
        elif shutil.which('aria2c'):
            subprocess.run(['aria2c', '-x', '5', '-j', ' 5',  '-q', src_url, '-d', cache_dir], check=True)
        elif shutil.which('curl'):
            subprocess.run(['curl', '-s', src_url, '-o', local_zip_file], check=True)
        else:
            import urllib.request
            urllib.request.urlretrieve(src_url, local_zip_file)
        if not check_md5(local_zip_file, expected_md5):
            print('Download failed')
            exit(-1)
    if os.path.exists(models_dir):
        print('deleting %s' % models_dir)
        shutil.rmtree(models_dir)
    if is_windows():
        subprocess.run(['powershell', '-Command', 'Expand-Archive -LiteralPath "%s" -DestinationPath "%s" -Force' %
                        (local_zip_file, models_dir)], check=True)
    else:
        subprocess.run(['unzip', '-qd', models_dir, local_zip_file], check=True)
    return True


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    return parser.parse_args()


args = parse_arguments()
os.makedirs(args.build_dir, exist_ok=True)

download_test_data(os.path.join(args.build_dir, 'cmake_temp'), test_data_url, test_data_checksum)
dest_dir = os.path.join(args.build_dir, 'cmake')
if os.path.exists(dest_dir):
    print('deleting %s' % dest_dir)
    shutil.rmtree(dest_dir)
shutil.move(os.path.join(args.build_dir, 'cmake_temp', 'cmake-3.13.2-win64-x64'), dest_dir)
if not os.path.exists(os.path.join(dest_dir, 'bin', 'cmake.exe')):
    print('download failed')
    exit(-1)
