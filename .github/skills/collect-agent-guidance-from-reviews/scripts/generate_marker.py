#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone


MARKER_PREFIX = "Agent-Guidance-Collection: version=1;"


def parse_utc_timestamp(value: str) -> datetime:
    if not value.endswith("Z"):
        raise argparse.ArgumentTypeError("timestamp must use UTC and end with 'Z'")

    try:
        timestamp = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO 8601 timestamp: {value}") from error

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an agent-guidance collection PR marker.")
    parser.add_argument("--since", required=True, type=parse_utc_timestamp)
    parser.add_argument(
        "--through",
        type=parse_utc_timestamp,
        help="UTC collection cutoff. Defaults to the current time.",
    )
    args = parser.parse_args()
    through = args.through or datetime.now(timezone.utc).replace(microsecond=0)

    if args.since >= through:
        parser.error("--since must be earlier than --through")

    print(generate_marker(args.since, through))


if __name__ == "__main__":
    main()
