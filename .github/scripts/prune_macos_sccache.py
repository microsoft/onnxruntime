# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Enforce the archived macOS sccache budget; default to a read-only preview."""

import argparse
import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

BUDGET_BYTES = 30_000_000_000
MAIN_REF = "refs/heads/main"
PILOT_REF = "refs/heads/eserscor-dev/erscor/2026/10/30-mac-compiler-cachi"
MAIN_KEY = re.compile(r"(sccache-v1-macOS-(?:ARM64|X64)-[a-f0-9]{64}-)[a-f0-9]{40}")
PILOT_KEY = re.compile(r"macos-sccache-v[12]-.+-[a-f0-9]{64}-[a-f0-9]{40}")


class Cache(TypedDict):
    id: int
    key: str
    ref: str
    size_in_bytes: int
    created_at: str


@dataclass
class EvictionPlan:
    total_bytes: int
    remaining_bytes: int
    deletions: list[Cache]
    evicted_baselines: list[Cache]


def compatibility(cache: Cache) -> str | None:
    if cache["ref"] == MAIN_REF:
        match = MAIN_KEY.fullmatch(cache["key"])
        if match:
            return match[1]
    return None


def in_scope(cache: Cache) -> bool:
    return bool(compatibility(cache) or (cache["ref"] == PILOT_REF and PILOT_KEY.fullmatch(cache["key"])))


def creation_time(cache: Cache) -> datetime:
    timestamp = datetime.fromisoformat(cache["created_at"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError(f"Cache timestamp must have a timezone: {cache['key']}")
    return timestamp


def select_evictions(inventory: Sequence[Cache], budget: int = BUDGET_BYTES) -> EvictionPlan:
    if type(budget) is not int or budget < 0:
        raise ValueError("Cache budget must be a nonnegative integer")
    caches = [cache for cache in inventory if in_scope(cache)]
    for cache in caches:
        if (
            type(cache["id"]) is not int
            or cache["id"] <= 0
            or type(cache["size_in_bytes"]) is not int
            or cache["size_in_bytes"] < 0
        ):
            raise ValueError(f"Invalid cache metadata for {cache['key']}")
        creation_time(cache)
    caches.sort(key=lambda cache: (creation_time(cache), cache["id"]), reverse=True)
    baselines = set()
    prefixes = set()
    for cache in caches:
        prefix = compatibility(cache)
        if prefix and prefix not in prefixes:
            prefixes.add(prefix)
            baselines.add(cache["id"])

    def priority(cache: Cache) -> int:
        # Retire pilot snapshots first, then old main generations, and only then main baselines.
        if cache["id"] in baselines:
            return 2
        return 1 if compatibility(cache) else 0

    candidates = sorted(caches, key=lambda cache: (priority(cache), creation_time(cache), cache["id"]))
    total_bytes = sum(cache["size_in_bytes"] for cache in caches)
    remaining_bytes = total_bytes
    deletions = []
    for cache in candidates:
        if remaining_bytes <= budget:
            break
        deletions.append(cache)
        remaining_bytes -= cache["size_in_bytes"]
    return EvictionPlan(
        total_bytes, remaining_bytes, deletions, [cache for cache in deletions if cache["id"] in baselines]
    )


def gh_api(args: list[str]) -> str:
    return subprocess.check_output(["gh", "api", *args], text=True, encoding="utf-8")


def list_caches(repository: str, api: Callable[[list[str]], str]) -> list[Cache]:
    caches = {}
    for key in ["sccache-v1-macOS-", "macos-sccache-v"]:
        pages = json.loads(api(["--paginate", "--slurp", f"repos/{repository}/actions/caches?per_page=100&key={key}"]))
        if not isinstance(pages, list) or not pages:
            raise ValueError("Invalid cache pagination response")
        for page in pages:
            if not isinstance(page, dict) or not isinstance(page.get("actions_caches"), list):
                raise ValueError("Missing actions_caches in API response")
            for cache in page["actions_caches"]:
                caches[cache["id"]] = cache
    return list(caches.values())


def run(
    repository: str,
    *,
    apply: bool = False,
    api: Callable[[list[str]], str] = gh_api,
    log: Callable[[str], None] = print,
) -> EvictionPlan:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository, flags=re.ASCII):
        raise ValueError("Expected repository in owner/name format")
    plan = select_evictions(list_caches(repository, api))
    log(
        f"{'Apply' if apply else 'Dry run'}: {plan.total_bytes} bytes; budget {BUDGET_BYTES}; "
        f"{len(plan.deletions)} evictions; planned remainder {plan.remaining_bytes} bytes"
    )
    if plan.evicted_baselines:
        log(f"::warning::Budget requires evicting {len(plan.evicted_baselines)} newest compatibility baselines")
    for cache in plan.deletions:
        # IDs are immutable; scope is checked again at the destructive boundary.
        if not in_scope(cache) or type(cache["id"]) is not int or cache["id"] <= 0:
            raise ValueError("Refusing to delete an out-of-scope cache")
        log(
            f"{'Deleting' if apply else 'Would delete'} cache {cache['id']}: "
            f"{cache['ref']} {cache['key']} ({cache['size_in_bytes']} bytes)"
        )
        if apply:
            api(["--method", "DELETE", f"repos/{repository}/actions/caches/{cache['id']}"])
    if apply:
        observed = select_evictions(list_caches(repository, api))
        log(f"Observed storage after cleanup: {observed.total_bytes} bytes")
        if observed.total_bytes > BUDGET_BYTES:
            raise RuntimeError(
                "macOS sccache remains over budget (possibly concurrent uploads); next maintenance run will retry"
            )
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete selected caches (default: dry run)")
    args = parser.parse_args()
    run(repository=os.environ["GITHUB_REPOSITORY"], apply=args.apply)
