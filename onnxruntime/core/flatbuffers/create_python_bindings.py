#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import subprocess


def update_namespace():
    # create a copy of the schema so we can replace the namespace so that the generated module name doesn't clash
    # with the 'onnxruntime' package.
    with open('schema/ort.fbs', 'r') as input, open('schema/ort.py.fbs', 'w') as output:
        for line in input:
            # convert any line with the namespace to use ort_flatbuffers_py instead of onnxruntime as the top level
            # namespace. this doesn't change how anything works - it just avoids a naming clash with the 'real'
            # onnxruntime python package
            output.write(line.replace('onnxruntime.experimental.fbs', 'ort_flatbuffers_py.experimental.fbs'))


def generate_schema(flatc):
    # run flatc to generate schema
    cmd = [flatc, '--python', 'schema/ort.py.fbs']
    subprocess.run(cmd, check=True)


def create_init_py():
    # create an __init__.py that imports all the py files so we can just 'import ort_flatbuffers_py.experimental.fbs'
    # in a script that wants to process an ORT format model
    os.chdir('ort_flatbuffers_py/experimental/fbs')
    with open('__init__.py', 'w') as init_py:
        init_py.write('''from os.path import dirname, basename, isfile, join, splitext
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [splitext(basename(f))[0] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from . import *  # noqa
''')


def main():
    parser = argparse.ArgumentParser(description='Generate python bindings for the ORT flatbuffers schema.',
                                     usage='Provide the path to the flatbuffers flatc executable. '
                                           'Script can be executed from anywhere but must be located in its original '
                                           'directory in the ONNX Runtime enlistment.')
    parser.add_argument('-f', '--flatc', required=True,
                        help='Path to flatbuffers flatc executable. '
                             'Can be found in the build directory under the external/flatbuffers/<config>/')

    args = parser.parse_args()

    # cd to script dir as everything we do is relative to that location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    update_namespace()
    generate_schema(args.flatc)
    create_init_py()


if __name__ == '__main__':
    main()
