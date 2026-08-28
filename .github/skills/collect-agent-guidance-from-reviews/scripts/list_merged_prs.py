#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_marker import MARKER_PREFIX, format_utc_timestamp, generate_marker, parse_utc_timestamp


AUDIT_MARKER_PREFIX = "Agent-Guidance-Audit: version=1;"
MARKER_PATTERN = re.compile(
    rf"^{re.escape(MARKER_PREFIX)} base=main; "
    r"harvested-since=(?P<since>\S+); harvested-through=(?P<through>\S+)\s*$",
    re.MULTILINE,
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


def run_gh(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["gh", *arguments],
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
    run_gh(["--version"])
    run_gh(["auth", "status"])


def resolve_repository(explicit_repository: str | None) -> str:
    if explicit_repository:
        return explicit_repository

    return run_gh(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]).strip()


def search_pull_requests(repository: str, query: str) -> list[dict[str, Any]]:
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


def discover_since(repository: str) -> datetime:
    marker_prs = search_pull_requests(
        repository,
        'is:pr is:merged base:main in:body "Agent-Guidance-Collection"',
    )
    through_values: list[datetime] = []

    for pull_request in marker_prs:
        for match in MARKER_PATTERN.finditer(pull_request.get("body") or ""):
            since = parse_utc_timestamp(match.group("since"))
            through = parse_utc_timestamp(match.group("through"))
            if since >= through:
                continue
            through_values.append(through)

    if not through_values:
        raise SystemExit("no valid merged collection marker found; pass --since for the initial collection")

    return max(through_values)


def list_candidates(repository: str, since: datetime, through: datetime) -> list[dict[str, Any]]:
    date_range = f"{since.date().isoformat()}..{through.date().isoformat()}"
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
        if MARKER_PREFIX in body or AUDIT_MARKER_PREFIX in body:
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
    parser.add_argument("--repo", help="GitHub repository in OWNER/REPO form. Defaults to the current repository.")
    parser.add_argument("--since", type=parse_utc_timestamp, help="UTC lower bound. Defaults to the latest marker.")
    parser.add_argument("--through", type=parse_utc_timestamp, help="UTC upper bound. Defaults to the current time.")
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout.")
    args = parser.parse_args()

    verify_gh()
    repository = resolve_repository(args.repo)
    through = args.through or datetime.now(timezone.utc).replace(microsecond=0)
    since = args.since or discover_since(repository)
    if since >= through:
        parser.error("--since must be earlier than --through")

    output = {
        "repository": repository,
        "base": "main",
        "since": format_utc_timestamp(since),
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
