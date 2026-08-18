# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Cherry-Pick Helper Script
-------------------------
Description:
    This script automates the process of cherry-picking commits for a release branch.
    It fetches merged PRs with a specific label, sorts them by merge date, and generates:
    1. A batch file (.cmd) with git cherry-pick commands.
    2. A markdown file (.md) for the PR description.
    It also checks for potential missing dependencies (conflicts) by verifying if files modified
    by the cherry-picked commits have any other modifications in commits that are not in the
    specified target branch and are not included in the cherry-pick list.

Usage:
    python cherry_pick.py --label "release:1.24.2" --output cherry_pick.cmd --branch "origin/rel-1.24.2"

Requirements:
    - Python 3.7+
    - GitHub CLI (gh) logged in.
    - Git available in PATH.
"""

import argparse
import json
import os
import sys

from cherry_pick_utils import (
    check_preflight,
    extract_pr_numbers,
    get_pr_number_from_subject,
    run_command,
)


def get_merged_prs(repo, label, limit=200):
    """Fetch merged PRs with the specific label."""
    print(f"Fetching merged PRs with label '{label}' from {repo}...")
    cmd = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--label",
        label,
        "--state",
        "merged",
        "--json",
        "number,title,mergeCommit,mergedAt",
        "-L",
        str(limit),
    ]
    output = run_command(cmd)
    if not output:
        return []

    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        print(f"Error parsing gh output: {e}", file=sys.stderr)
        return []


def get_changed_files(oid):
    """Get list of files changed in a commit."""
    output = run_command(["git", "diff-tree", "--no-commit-id", "--name-only", "-m", "-r", oid], silent=True)
    if output:
        return output.strip().splitlines()
    return []


def sanitize_title(title):
    """Normalize PR titles for single-line text output."""
    return title.replace("\n", " ").strip()


def escape_markdown_table_cell(text):
    """Escape markdown table delimiters in cell content."""
    return sanitize_title(text).replace("|", "\\|")


def get_existing_pr_numbers(branch, repo=None, log_limit=500):
    """Get the set of PR numbers already present in the target branch."""
    output = run_command(["git", "log", branch, "--oneline", "-n", str(log_limit)], silent=True)
    if not output:
        return set()
    pr_numbers = set()

    # Pre-fetch PR cache to avoid redundant gh calls
    pr_cache = {}

    # Process commit log
    lines = output.strip().splitlines()
    for line in lines:
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        subject = parts[1]

        pr_num = get_pr_number_from_subject(subject)
        if not pr_num:
            continue

        pr_num_int = int(pr_num)
        pr_numbers.add(pr_num_int)

        # Check if it's a cherry-pick / meta-PR
        is_meta_pr = (
            "cherry pick" in subject.lower() or "cherry-pick" in subject.lower() or "cherrypick" in subject.lower()
        )

        if is_meta_pr:
            # Query gh to get more details (body/commits) to find squashed sub-PRs
            if pr_num not in pr_cache:
                gh_cmd = ["gh", "pr", "view", pr_num, "--json", "title,body,commits"]
                if repo:
                    gh_cmd.extend(["--repo", repo])
                gh_out = run_command(gh_cmd, silent=True)
                if gh_out:
                    try:
                        pr_cache[pr_num] = json.loads(gh_out)
                    except json.JSONDecodeError:
                        pr_cache[pr_num] = None
                else:
                    pr_cache[pr_num] = None

            details = pr_cache.get(pr_num)
            if details:
                # Collect sub-PRs from title, body, and commits
                extracted_nums = []
                extracted_nums.extend(extract_pr_numbers(details.get("title", ""), strict=True))
                extracted_nums.extend(extract_pr_numbers(details.get("body", ""), strict=True))

                for commit in details.get("commits", []):
                    extracted_nums.extend(extract_pr_numbers(commit.get("messageHeadline", ""), strict=True))

                for num in set(extracted_nums):
                    if num != pr_num_int:
                        pr_numbers.add(num)

    return pr_numbers


def check_missing_dependencies(prs, branch):
    """Check for potential missing dependencies (conflicts)."""
    print("\nChecking for potential missing dependencies (conflicts)...")

    # Collect OIDs being cherry-picked and all their ancestor commits
    cherry_pick_oids = set()
    for pr in prs:
        if pr.get("mergeCommit"):
            merge_oid = pr["mergeCommit"]["oid"]
            cherry_pick_oids.add(merge_oid)
            # Include ancestor commits of merge commits to avoid false-positive warnings
            # for PRs that used a regular merge (not squash) strategy
            ancestor_output = run_command(["git", "log", "--format=%H", merge_oid, "--not", branch], silent=True)
            if ancestor_output:
                for ancestor_oid in ancestor_output.strip().splitlines():
                    cherry_pick_oids.add(ancestor_oid.strip())

    conflicting_prs_count = 0
    for pr in prs:
        if not pr.get("mergeCommit"):
            continue

        oid = pr["mergeCommit"]["oid"]
        number = pr["number"]

        files = get_changed_files(oid)
        if not files:
            continue

        # For each file, find commits that modified it between the target branch and the cherry-picked commit.
        # Deduplicate warnings: group affected files by missing commit.
        # missing_commits maps: missing_commit_oid -> {"title": ..., "files": [...]}
        missing_commits = {}

        for filepath in files:
            # git log <cherry-pick-commit> --not <target-branch> -- <file>
            output = run_command(["git", "log", oid, "--not", branch, "--format=%H %s", "--", filepath], silent=True)

            if not output:
                continue

            for line in output.strip().splitlines():
                parts = line.split(" ", 1)
                c = parts[0]
                title = parts[1] if len(parts) > 1 else ""

                if c == oid:
                    continue
                if c not in cherry_pick_oids:
                    entry = missing_commits.setdefault(c, {"title": title, "files": []})
                    if not entry["title"]:
                        entry["title"] = title
                    entry["files"].append(filepath)

        # Print deduplicated warnings
        if missing_commits:
            conflicting_prs_count += 1
            for missing_oid, entry in missing_commits.items():
                files_str = ", ".join(entry["files"])
                print(
                    f"WARNING: PR #{number} ({oid}) modifies files that were also changed by commit {missing_oid} ({entry['title']}), "
                    f"which is not in the cherry-pick list. This may indicate missing related changes. Affected files: {files_str}"
                )

    if conflicting_prs_count == 0:
        print("No potential missing dependencies found.")
    else:
        print(f"\nDone. Found potential missing dependencies for {conflicting_prs_count} PRs.")


def main():
    parser = argparse.ArgumentParser(description="Generate cherry-pick script from PRs with a specific label.")
    parser.add_argument("--label", required=True, help="Label to filter PRs")
    parser.add_argument(
        "--output", required=True, help="Output script file path (.sh for bash, .cmd for Windows batch)"
    )
    parser.add_argument("--repo", default="microsoft/onnxruntime", help="Repository (default: microsoft/onnxruntime)")
    parser.add_argument(
        "--branch", default="HEAD", help="Target branch to compare against for dependency checks (default: HEAD)"
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of PRs to fetch (default: 200)")
    parser.add_argument(
        "--md-output",
        help="Output markdown file path for the PR description (default: next to --output)",
    )
    args = parser.parse_args()

    # Preflight Check
    if not check_preflight():
        return

    # 1. Fetch Merged PRs
    prs = get_merged_prs(args.repo, args.label, args.limit)

    if not prs:
        print(f"No PRs found with label '{args.label}'.")
        return

    # Sort by mergedAt (ISO 8601 strings sort correctly in chronological order)
    prs.sort(key=lambda x: x["mergedAt"])

    # 1.5. Check which PRs are already in the target branch
    existing_prs = get_existing_pr_numbers(args.branch, repo=args.repo)
    if existing_prs:
        print(f"Found {len(existing_prs)} PRs already in branch '{args.branch}'.")

    cherry_pick_prs = []
    skipped_count = 0
    for pr in prs:
        number = pr["number"]
        safe_title = sanitize_title(pr["title"])

        if not pr.get("mergeCommit"):
            print(f"Warning: PR #{number} has no merge commit OID. Skipping.", file=sys.stderr)
            continue

        if number in existing_prs:
            print(f"Skipping PR #{number} (already in branch '{args.branch}'): {safe_title}")
            skipped_count += 1
            continue

        cherry_pick_prs.append(pr)

    # Determine output format based on file extension
    is_shell = args.output.endswith(".sh")

    # 2. Write Output Script
    commit_count = len(cherry_pick_prs)
    with open(args.output, "w", encoding="utf-8") as f:
        if is_shell:
            f.write("#!/bin/bash\n")
            f.write(f"# Cherry-pick {args.label} commits\n")
            f.write("# Sorted by merge time (oldest first)\n")
            f.write("set -e\n\n")
        else:
            f.write("@echo off\n")
            f.write(f"rem Cherry-pick {args.label} commits\n")
            f.write("rem Sorted by merge time (oldest first)\n\n")

        for pr in cherry_pick_prs:
            number = pr["number"]
            safe_title = sanitize_title(pr["title"])

            oid = pr["mergeCommit"]["oid"]
            comment = "#" if is_shell else "rem"
            f.write(f"{comment} PR {number}: {safe_title}\n")
            f.write(f"git cherry-pick {oid}\n\n")

    print(f"Generated {args.output} with {commit_count} commits ({skipped_count} skipped, already in branch).")

    # 3. Write PR Description Markdown (table format)
    output_dir = os.path.dirname(args.output)
    md_output = args.md_output or os.path.join(output_dir, "cherry_pick_pr_description.md")
    with open(md_output, "w", encoding="utf-8") as f:
        f.write("This cherry-picks the following commits for the release:\n\n")
        f.write("| Commit ID | PR Number | Commit Title |\n")
        f.write("|-----------|-----------|-------------|\n")
        for pr in cherry_pick_prs:
            number = pr["number"]
            title = escape_markdown_table_cell(pr["title"])
            oid = pr["mergeCommit"]["oid"]
            short_oid = oid[:10]
            f.write(f"| {short_oid} | #{number} | {title} |\n")

    print(f"Generated {md_output} with {commit_count} commits.")

    # 4. Dependency Check
    check_missing_dependencies(cherry_pick_prs, args.branch)


if __name__ == "__main__":
    main()
