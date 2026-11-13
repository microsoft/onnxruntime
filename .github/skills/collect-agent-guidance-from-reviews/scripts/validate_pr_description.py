#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

from collection_marker import generate_marker, parse_marker, parse_utc_timestamp


def validate_pr_description(body: str, collection_output: dict[str, Any]) -> None:
    """Validates that a PR description contains the marker for a frozen collection run."""
    try:
        since_text = collection_output["since"]
        through_text = collection_output["through"]
        recorded_marker = collection_output["marker"]
    except KeyError as error:
        raise ValueError(f"collection output is missing the {error.args[0]!r} field") from error

    if not all(isinstance(value, str) for value in (since_text, through_text, recorded_marker)):
        raise ValueError("collection output fields must be strings")

    expected_since = parse_utc_timestamp(since_text)
    expected_through = parse_utc_timestamp(through_text)
    expected_marker = generate_marker(expected_since, expected_through)
    if recorded_marker != expected_marker:
        raise ValueError("collection output marker does not match its since and through fields")

    actual_since, actual_through = parse_marker(body)
    if (actual_since, actual_through) != (expected_since, expected_through):
        raise ValueError(
            f"PR description marker does not match the frozen collection window: expected {expected_marker}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a collection PR description and marker.")
    parser.add_argument("--body-file", required=True, type=Path, help="Proposed PR description.")
    parser.add_argument(
        "--collection-output",
        required=True,
        type=Path,
        help="JSON output preserved from list_merged_prs.py.",
    )
    args = parser.parse_args()

    body = args.body_file.read_text(encoding="utf-8")
    collection_output = json.loads(args.collection_output.read_text(encoding="utf-8"))
    if not isinstance(collection_output, dict):
        raise ValueError("collection output must be a JSON object")
    validate_pr_description(body, collection_output)


if __name__ == "__main__":
    main()
