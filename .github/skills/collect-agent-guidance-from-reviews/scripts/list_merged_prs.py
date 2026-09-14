#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collection_marker import (
    MARKER_PATTERN,
    format_utc_timestamp,
    generate_marker,
    parse_marker,
    parse_utc_timestamp,
)

SEARCH_QUERY = """
query($searchQuery: String!, $endCursor: String) {
  search(query: $searchQuery, type: ISSUE, first: 100, after: $endCursor) {
    issueCount
    nodes {
      ... on PullRequest {
        number
        title
        url
        body
        mergedAt
        baseRefName
        author {
          login
        }
        mergeCommit {
          oid
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""


def run_gh(arguments: list[str], cwd: Path | None = None) -> str:
    """Runs an authenticated GitHub CLI command and returns its standard output."""
    try:
        result = subprocess.run(
            ["gh", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        raise SystemExit("gh was not found on PATH") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or str(error)
        raise SystemExit(f"gh command failed: {message}") from error

    return result.stdout


def verify_gh() -> None:
    """Verifies that the GitHub CLI is installed and authenticated."""
    run_gh(["--version"])
    run_gh(["auth", "status"])


def resolve_repository() -> str:
    """Resolves the repository containing this script, independent of the caller's directory."""
    script_directory = Path(__file__).resolve().parent
    return run_gh(
        ["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
        cwd=script_directory,
    ).strip()


def parse_timestamp_argument(value: str) -> datetime:
    """Parses a date or timezone-aware ISO 8601 timestamp and normalizes it to UTC."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        except ValueError as error:
            raise ValueError(f"invalid ISO 8601 date: {value}") from error

    try:
        normalized_value = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        timestamp = datetime.fromisoformat(normalized_value)
    except ValueError as error:
        raise ValueError(f"invalid ISO 8601 timestamp: {value}") from error
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset or end with 'Z'")

    return timestamp.astimezone(timezone.utc)


def search_pull_requests(repository: str, query: str) -> list[dict[str, Any]]:
    """Returns all PRs matching a GitHub search query, up to GitHub's 1,000-result limit."""
    nodes: list[dict[str, Any]] = []
    end_cursor: str | None = None

    while True:
        arguments = [
            "api",
            "graphql",
            "-f",
            f"query={SEARCH_QUERY}",
            "-f",
            f"searchQuery=repo:{repository} {query}",
        ]
        if end_cursor is not None:
            arguments.extend(["-f", f"endCursor={end_cursor}"])

        response = json.loads(run_gh(arguments))
        search = response["data"]["search"]
        if search["issueCount"] > 1000:
            raise SystemExit("GitHub search returned more than 1,000 PRs; use a narrower collection window")

        nodes.extend(node for node in search["nodes"] if node is not None)
        page_info = search["pageInfo"]
        if not page_info["hasNextPage"]:
            return nodes

        end_cursor = page_info["endCursor"]


def warn_invalid_marker(pull_request: dict[str, Any], reason: str) -> None:
    """Reports rejected cursor metadata without contaminating JSON output."""
    print(
        f"warning: ignoring collection marker in PR #{pull_request['number']}: {reason}",
        file=sys.stderr,
    )


def discover_since(repository: str, through: datetime, warn_about_later_markers: bool = False) -> tuple[datetime, int]:
    """Finds the latest valid collection cutoff and the merged PR that recorded it."""
    marker_prs = search_pull_requests(
        repository,
        'is:pr is:merged base:main in:body "Agent-Guidance-Collection"',
    )
    cursors: list[tuple[datetime, int]] = []

    for pull_request in marker_prs:
        matches = list(MARKER_PATTERN.finditer(pull_request.get("body") or ""))
        # The GitHub text search also returns PRs that mention the marker name without containing marker metadata.
        if not matches:
            continue

        try:
            _, harvested_through = parse_marker(pull_request.get("body") or "")
        except ValueError as error:
            warn_invalid_marker(pull_request, str(error))
            continue
        if harvested_through > through:
            if warn_about_later_markers:
                warn_invalid_marker(
                    pull_request,
                    f"harvested-through {format_utc_timestamp(harvested_through)} is later than "
                    f"the collection cutoff {format_utc_timestamp(through)}",
                )
            continue

        cursors.append((harvested_through, pull_request["number"]))

    if not cursors:
        raise SystemExit("no valid merged collection marker found; pass --since for the initial collection")

    return max(cursors, key=lambda cursor: (cursor[0], cursor[1]))


def list_candidates(repository: str, since: datetime, through: datetime) -> list[dict[str, Any]]:
    """Lists merged main-targeting PRs in the half-open collection window."""
    date_range = f"{format_utc_timestamp(since)}..{format_utc_timestamp(through)}"
    pull_requests = search_pull_requests(repository, f"is:pr is:merged base:main merged:{date_range}")
    candidates: list[dict[str, Any]] = []

    for pull_request in pull_requests:
        merged_at_text = pull_request.get("mergedAt")
        if not merged_at_text:
            continue

        merged_at = parse_utc_timestamp(merged_at_text)
        if not since < merged_at <= through or pull_request.get("baseRefName") != "main":
            continue

        body = pull_request.get("body") or ""
        try:
            parse_marker(body)
        except ValueError:
            pass
        else:
            continue

        candidates.append(
            {
                "number": pull_request["number"],
                "url": pull_request["url"],
                "title": pull_request["title"],
                "author": (pull_request.get("author") or {}).get("login"),
                "merged_at": format_utc_timestamp(merged_at),
                "merge_commit": (pull_request.get("mergeCommit") or {}).get("oid"),
            }
        )

    return sorted(candidates, key=lambda pull_request: (pull_request["merged_at"], pull_request["number"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="List merged main-targeting PRs for agent-guidance collection.")
    parser.add_argument("--since", type=parse_timestamp_argument, help="Lower bound. Defaults to the latest marker.")
    parser.add_argument("--through", type=parse_timestamp_argument, help="Upper bound. Defaults to the current time.")
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout.")
    args = parser.parse_args()

    verify_gh()
    repository = resolve_repository()
    through = args.through or datetime.now(timezone.utc).replace(microsecond=0)
    collection_start_source_pr: int | None = None
    if args.since:
        since = args.since
    else:
        since, collection_start_source_pr = discover_since(
            repository,
            through,
            warn_about_later_markers=args.through is None,
        )
    if since >= through:
        parser.error("--since must be earlier than --through")

    output = {
        "repository": repository,
        "base": "main",
        "since": format_utc_timestamp(since),
        "collection_start_source_pr": collection_start_source_pr,
        "through": format_utc_timestamp(through),
        "marker": generate_marker(since, through),
        "pull_requests": list_candidates(repository, since, through),
    }
    serialized_output = f"{json.dumps(output, indent=2, sort_keys=True)}\n"

    if args.output:
        args.output.write_text(serialized_output, encoding="utf-8")
    else:
        print(serialized_output, end="")


if __name__ == "__main__":
    main()
