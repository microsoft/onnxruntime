import logging

from .util import Colors, on_ci

_INITIALIZED = False


class ColorLogFormatter(logging.Formatter):
    FORMATTERS_BY_LEVEL = {
        logging.DEBUG: logging.Formatter(f"{Colors.GREY}[bnt] %(message)s{Colors.OFF}"),
        logging.INFO: logging.Formatter("[bnt] %(message)s"),
        logging.WARN: logging.Formatter(
            f"{Colors.YELLOW}[bnt] %(message)s{Colors.OFF}"
        ),
        logging.ERROR: logging.Formatter(
            f"{Colors.RED_BOLD}[bnt] %(message)s{Colors.OFF}"
        ),
        logging.CRITICAL: logging.Formatter(
            f"{Colors.RED_REVERSED_VIDEO}[bnt] %(message)s{Colors.OFF}"
        ),
        logging.FATAL: logging.Formatter(
            f"{Colors.RED_REVERSED_VIDEO}[bnt] %(message)s{Colors.OFF}"
        ),
    }
    DEFAULT_FORMATTER = FORMATTERS_BY_LEVEL[logging.ERROR]

    def format(self, record: logging.LogRecord) -> str:
        return self.FORMATTERS_BY_LEVEL.get(
            record.levelno, self.DEFAULT_FORMATTER
        ).format(record)


def initialize_logging() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    if on_ci():
        CI_LOG_FORMAT = "[%(asctime)s] [bnt] [%(levelname)s] %(message)s"
        logging.basicConfig(level=logging.DEBUG, format=CI_LOG_FORMAT, force=True)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler()
        handler.setFormatter(ColorLogFormatter())
        logger.addHandler(handler)

        # The backoff library we use for retries doesn't log its retries by default. This turns on logging.
        logging.getLogger("backoff").addHandler(handler)
