#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys

import semver

input = sys.argv[1]

output = input.strip()

if output != '':

  ver = semver.Version.parse(output)

  if ver.prerelease:
    prefix = ver.prerelease.split('.')[0]
    if not prefix in ('alpha', 'beta', 'rc'):
        raise ValueError(f"Invalid pre-release: {ver}. (alpha|beta|rc) accepted.")

print(f'##vso[task.setvariable variable={ReleaseVersionSuffix};]{output}')
