# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Usage:
# python create_cherry_pick.py --label "release:1.24.2" --output cherry_pick.cmd --branch "origin/rel-1.24.2"
#
# Arguments:
#   --label: Label to filter PRs (required)
#   --output: Output cmd file path (required)
#   --repo: Repository (default: microsoft/onnxruntime)
#   --branch: Target branch to compare against for dependency checks (default: HEAD)
#
# This script fetches merged PRs with the specified label from the onnxruntime repository,
# sorts them by merge date, and generates:
# 1. A batch file (specified by --output) containing git cherry-pick commands.
# 2. A markdown file (cherry_pick_pr_description.md) summarizing the cherry-picked PRs for pull request description.
#
# It also checks for potential missing dependencies (conflicts) by verifying if files modified
# by the cherry-picked commits have any other modifications in the target branch history
# that are not included in the cherry-pick list.
import argparse
import json
import subprocess
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Generate cherry-pick script from PRs with a specific label.")
    parser.add_argument("--label", required=True, help="Label to filter PRs")
    parser.add_argument("--output", required=True, help="Output cmd file path")
    parser.add_argument("--repo", default="microsoft/onnxruntime", help="Repository (default: microsoft/onnxruntime)")
    parser.add_argument(
        "--branch", default="HEAD", help="Target branch to compare against for dependency checks (default: HEAD)"
    )
    args = parser.parse_args()

    # Fetch merged PRs with the specified label using gh CLI
    print(f"Fetching merged PRs with label '{args.label}' from {args.repo}...")
    cmd = [
        "gh",
        "pr",
        "list",
        "--repo",
        args.repo,
        "--label",
        args.label,
        "--state",
        "merged",
        "--json",
        "number,title,mergeCommit,mergedAt",
        "-L",
        "200",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        prs = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running gh command: {e}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing gh output: {e}", file=sys.stderr)
        print(f"Output was: {result.stdout}", file=sys.stderr)
        sys.exit(1)

    if not prs:
        print(f"No PRs found with label '{args.label}'.")
        return

    # Sort by mergedAt (ISO 8601 strings sort correctly in chronological order)
    prs.sort(key=lambda x: x["mergedAt"])

    # Write to output cmd file
    commit_count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write(f"rem Cherry-pick {args.label} commits\n")
        f.write("rem Sorted by merge time (oldest first)\n\n")

        for pr in prs:
            number = pr["number"]
            title = pr["title"]
            safe_title = title.replace("\n", " ")

            if not pr.get("mergeCommit"):
                print(f"Warning: PR #{number} has no merge commit OID. Skipping.", file=sys.stderr)
                continue

            oid = pr["mergeCommit"]["oid"]
            f.write(f"rem PR {number}: {safe_title}\n")
            f.write(f"git cherry-pick {oid}\n\n")
            commit_count += 1

    print(f"Generated {args.output} with {commit_count} commits.")

    # Write to markdown file. You can use it as the pull request description.
    md_output = "cherry_pick_pr_description.md"
    with open(md_output, "w", encoding="utf-8") as f:
        f.write("This cherry-picks the following commits for the release:\n")
        for pr in prs:
            if not pr.get("mergeCommit"):
                continue
            number = pr["number"]
            f.write(f"- #{number}\n")

    print(f"Generated {md_output} with {commit_count} commits.")

    # Check for potential missing dependencies
    print("\nChecking for potential missing dependencies (conflicts)...")

    # Collect OIDs being cherry-picked
    cherry_pick_oids = set()
    for pr in prs:
        if pr.get("mergeCommit"):
            cherry_pick_oids.add(pr["mergeCommit"]["oid"])

    for pr in prs:
        if not pr.get("mergeCommit"):
            continue

        oid = pr["mergeCommit"]["oid"]
        number = pr["number"]

        # Get files changed by this commit
        try:
            res = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", oid],
                capture_output=True,
                text=True,
                check=True,
            )
            files = res.stdout.strip().splitlines()
        except subprocess.CalledProcessError as e:
            print(f"Error getting changed files for {oid}: {e}", file=sys.stderr)
            continue

        # For each file, find commits that modified it between the target branch and the cherry-picked commit.
        # Deduplicate warnings: group affected files by missing commit.
        # missing_commits maps: missing_commit_oid -> (title, [list of affected files])
        missing_commits = defaultdict(lambda: ("", []))
        for filepath in files:
            try:
                res = subprocess.run(
                    ["git", "log", oid, "--not", args.branch, "--format=%H %s", "--", filepath],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                for line in res.stdout.strip().splitlines():
                    parts = line.split(" ", 1)
                    c = parts[0]
                    title = parts[1] if len(parts) > 1 else ""

                    if c == oid:
                        continue
                    if c not in cherry_pick_oids:
                        existing_title, existing_files = missing_commits[c]
                        if not existing_title:
                            existing_title = title
                        existing_files.append(filepath)
                        missing_commits[c] = (existing_title, existing_files)

            except subprocess.CalledProcessError as e:
                print(f"Error checking history for {filepath}: {e}", file=sys.stderr)
                continue

        # Print deduplicated warnings
        for missing_oid, (title, affected_files) in missing_commits.items():
            files_str = ", ".join(affected_files)
            print(
                f"WARNING: PR #{number} ({oid}) depends on commit {missing_oid} ({title}) "
                f"which is not in the cherry-pick list. Affected files: {files_str}"
            )


if __name__ == "__main__":
    main()
