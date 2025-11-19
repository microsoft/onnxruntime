#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os


def parse_qnn_version_from_sdk_yaml(qnn_home):
    sdk_file = os.path.join(qnn_home, "sdk.yaml")
    with open(sdk_file) as f:
        for line in f:
            if line.strip().startswith("version:"):
                # yaml file has simple key: value format with version as key
                return line.split(":", 1)[1].strip()
    return None
