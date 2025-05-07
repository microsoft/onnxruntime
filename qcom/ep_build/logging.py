# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
from collections.abc import Mapping
from typing import ClassVar

from .util import Colors, is_host_in_ci

_INITIALIZED = False


class ColorLogFormatter(logging.Formatter):
    FORMATTERS_BY_LEVEL: ClassVar[Mapping[int, logging.Formatter]] = {
        logging.DEBUG: logging.Formatter(f"{Colors.GREY}[bnt] %(message)s{Colors.OFF}"),
        logging.INFO: logging.Formatter("[bnt] %(message)s"),
        logging.WARN: logging.Formatter(f"{Colors.YELLOW}[bnt] %(message)s{Colors.OFF}"),
        logging.ERROR: logging.Formatter(f"{Colors.RED_BOLD}[bnt] %(message)s{Colors.OFF}"),
        logging.CRITICAL: logging.Formatter(f"{Colors.RED_REVERSED_VIDEO}[bnt] %(message)s{Colors.OFF}"),
        logging.FATAL: logging.Formatter(f"{Colors.RED_REVERSED_VIDEO}[bnt] %(message)s{Colors.OFF}"),
    }
    DEFAULT_FORMATTER = FORMATTERS_BY_LEVEL[logging.ERROR]

    def format(self, record: logging.LogRecord) -> str:
        return self.FORMATTERS_BY_LEVEL.get(record.levelno, self.DEFAULT_FORMATTER).format(record)


def initialize_logging() -> None:
    global _INITIALIZED  # noqa: PLW0603
    if _INITIALIZED:
        return
    _INITIALIZED = True

    if is_host_in_ci():
        ci_log_format = "[%(asctime)s] [bnt] [%(levelname)s] %(message)s"
        logging.basicConfig(level=logging.DEBUG, format=ci_log_format, force=True)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler()
        handler.setFormatter(ColorLogFormatter())
        logger.addHandler(handler)

        # The backoff library we use for retries doesn't log its retries by default. This turns on logging.
        logging.getLogger("backoff").addHandler(handler)
