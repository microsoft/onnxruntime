# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNX Runtime create nuget spec script",
                                     usage='')
    # Main arguments
    parser.add_argument("--source_dir", required=True, help="Path to the source directory.")
    parser.add_argument("--debug_binary_root", required=True, help="Path to the debug binary directory.")
    parser.add_argument("--release_binary_root", required=True, help="Path to the release binary directory.")
    return parser.parse_args()


def generate_nuspec(source_dir, debug_binary_root, release_binary_root, architecture):
    template_path = '%s/tools/nuget/template.nuspec' % source_dir
    with open(template_path, 'rt') as f:
        template = f.read()
        return template.replace('@@DebugBinaryRoot@@', debug_binary_root)\
            .replace('@@ReleaseBinaryRoot@@', release_binary_root)\
            .replace('@@MSBuildArchitecture@@', architecture)\
            .replace('@@SrcRoot@@', source_dir)


def generate_targets(source_dir):
    template_path = '%s/tools/nuget/template.targets' % source_dir
    with open(template_path, 'rt') as f:
        template = f.read()
        return template


def main():
    args = parse_arguments()
    nuspec = generate_nuspec(args.source_dir, args.debug_binary_root, args.release_binary_root, 'amd64')
    with open('onnxruntime.nuspec', 'wt') as f:
        f.write(nuspec)
    targets = generate_targets(args.source_dir)
    with open('onnxruntime.targets', 'wt') as f:
        f.write(targets)


if __name__ == "__main__":
    sys.exit(main())
