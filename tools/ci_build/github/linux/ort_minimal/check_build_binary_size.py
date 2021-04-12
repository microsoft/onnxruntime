#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import sys

# local helpers
import readelf_utils


def _check_binary_size(path, readelf, threshold, os_str, arch, build_config):

    print('Checking binary size of {} using {}'.format(path, readelf))
    ondisk_size = os.path.getsize(path)

    print('Section:size in bytes')
    # call get_section_sizes to dump the section info to stdout
    sections = readelf_utils.get_section_sizes(path, readelf, sys.stdout)
    sections_total = sum(sections.values())

    print('Sections total={} bytes'.format(sections_total))
    print('File size={} bytes'.format(ondisk_size))

    # Write the binary size to a file for uploading later
    # On-disk binary size jumps in 4KB increments so we use the total of the sections as it has finer granularity.
    # Note that the sum of the section is slightly larger than the on-disk size
    # due to packing and/or alignment adjustments.
    with open(os.path.join(os.path.dirname(path), 'binary_size_data.txt'), 'w') as file:
        file.writelines([
            'os,arch,build_config,size\n',
            '{},{},{},{}\n'.format(os_str, arch, build_config, sections_total)
        ])

    if threshold is not None and sections_total > threshold:
        raise RuntimeError('Sections total size for {} of {} exceeds threshold of {} by {}. On-disk size={}'
                           .format(path, sections_total, threshold, sections_total - threshold, ondisk_size))


def main():
    argparser = argparse.ArgumentParser(description='Check the binary size for provided path and '
                                                    'create a text file for upload to the performance dashboard.')

    # optional
    argparser.add_argument('-t', '--threshold', type=int,
                           help='Return error if binary size exceeds this threshold.')
    argparser.add_argument('-r', '--readelf_path', type=str, default='readelf', help='Path to readelf executable.')
    argparser.add_argument('--os', type=str, default='android',
                           help='OS value to include in binary_size_data.txt')
    argparser.add_argument('--arch', type=str, default='arm64-v8a',
                           help='Arch value to include in binary_size_data.txt')
    argparser.add_argument('--build_config', type=str, default='minimal-baseline',
                           help='Build_config value to include in binary_size_data.txt')

    # file to analyze
    argparser.add_argument('path', type=os.path.realpath, help='Path to binary to check.')

    args = argparser.parse_args()

    _check_binary_size(args.path, args.readelf_path, args.threshold, args.os, args.arch, args.build_config)


if __name__ == '__main__':
    main()
