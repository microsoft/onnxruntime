"""Collection marker schema, formatting, and parsing."""

import re
from datetime import datetime

MARKER_PREFIX = "Agent-Guidance-Collection: version=1;"
MARKER_PATTERN = re.compile(
    rf"^[ \t]*{re.escape(MARKER_PREFIX)} base=main; "
    r"harvested-since=(?P<since>[^ \t\r\n;]+); "
    r"harvested-through=(?P<through>[^ \t\r\n]+)[ \t]*\r?$",
    re.MULTILINE,
)


def parse_utc_timestamp(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError("timestamp must use UTC and end with 'Z'")

    try:
        timestamp = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise ValueError(f"invalid ISO 8601 timestamp: {value}") from error

    return timestamp


def format_utc_timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def generate_marker(since: datetime, through: datetime) -> str:
    if since >= through:
        raise ValueError("since must be earlier than through")

    return (
        f"{MARKER_PREFIX} base=main; "
        f"harvested-since={format_utc_timestamp(since)}; "
        f"harvested-through={format_utc_timestamp(through)}"
    )


def parse_marker(body: str) -> tuple[datetime, datetime]:
    """Parses the single collection marker required in a PR description."""
    matches = list(MARKER_PATTERN.finditer(body))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one marker line, found {len(matches)}")

    match = matches[0]
    since = parse_utc_timestamp(match.group("since"))
    through = parse_utc_timestamp(match.group("through"))
    if since >= through:
        raise ValueError(
            "harvested-since "
            f"({match.group('since')}) must be earlier than harvested-through ({match.group('through')})"
        )

    return since, through
