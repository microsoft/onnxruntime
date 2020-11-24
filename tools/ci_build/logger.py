# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging


def get_logger(name):
    logging.basicConfig(
        format="%(asctime)s %(name)s [%(levelname)s] - %(message)s",
        level=logging.DEBUG)

    return logging.getLogger(name)
