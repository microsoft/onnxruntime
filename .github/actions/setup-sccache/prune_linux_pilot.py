# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import subprocess

PREFIX = "sccache-linux-pilot-v1-Linux-X64-"
BUDGET_BYTES = 20 * 1024**3
MAX_SNAPSHOTS = 2


def eviction_plan(caches):
    snapshots = sorted(
        (cache for cache in caches if cache["ref"] == "refs/heads/main" and cache["key"].startswith(PREFIX)),
        key=lambda cache: (cache["created_at"], cache["id"]),
        reverse=True,
    )
    total = sum(cache["size_in_bytes"] for cache in snapshots)
    remaining = total
    victims = []
    # Keep the newest main baseline even if one archive alone exceeds the soft budget.
    for cache in reversed(snapshots[1:]):
        if remaining <= BUDGET_BYTES and len(snapshots) - len(victims) <= MAX_SNAPSHOTS:
            break
        victims.append(cache)
        remaining -= cache["size_in_bytes"]
    return victims, total, remaining


def run(repository, apply=False):
    pages = json.loads(
        subprocess.check_output(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{repository}/actions/caches?ref=refs/heads/main&key={PREFIX}&per_page=100",
            ],
            text=True,
            encoding="utf-8",
        )
    )
    victims, total, remaining = eviction_plan([cache for page in pages for cache in page["actions_caches"]])
    print(f"Linux pilot archived bytes: {total}; projected after cleanup: {remaining}; soft budget: {BUDGET_BYTES}")
    for cache in victims:
        print(f"{'Delete' if apply else 'Would delete'} {cache['id']}: {cache['key']} ({cache['size_in_bytes']} bytes)")
        if apply:
            subprocess.run(
                ["gh", "api", "--method", "DELETE", f"repos/{repository}/actions/caches/{cache['id']}"],
                check=True,
            )
    if remaining > BUDGET_BYTES:
        print("::warning::Newest Linux pilot baseline exceeds the soft archive budget; retained to avoid a cold reset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bound only Linux pilot main archives; default is dry-run.")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run(os.environ["GITHUB_REPOSITORY"], apply=args.apply)
