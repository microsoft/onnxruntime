#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s",
    level=logging.DEBUG)

log = logging.getLogger("Build")
