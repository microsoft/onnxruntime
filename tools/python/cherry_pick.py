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
import re
import subprocess
import sys
from collections import defaultdict


def run_command(command_list, cwd=None, silent=False):
    """Run a command using a list of arguments for security (no shell=True)."""
    try:
        result = subprocess.run(command_list, check=False, capture_output=True, text=True, cwd=cwd, encoding="utf-8")
        if result.returncode != 0:
            if not silent:
                log_str = " ".join(command_list)
                print(f"Error running command: {log_str}", file=sys.stderr)
                if result.stderr:
                    print(f"Stderr: {result.stderr.strip()}", file=sys.stderr)
            return None
        return result.stdout
    except FileNotFoundError:
        if not silent:
            cmd = command_list[0]
            print(f"Error: '{cmd}' command not found.", file=sys.stderr)
            if cmd == "gh":
                print(
                    "Please install GitHub CLI (https://cli.github.com/) and ensure 'gh' is available on your PATH.",
                    file=sys.stderr,
                )
        return None
    except Exception as e:
        if not silent:
            print(f"Exception running command {' '.join(command_list)}: {e}", file=sys.stderr)
        return None


def check_preflight():
    """Verify gh CLI and git repository early."""
    # Check git
    git_check = run_command(["git", "rev-parse", "--is-inside-work-tree"], silent=True)
    if not git_check:
        print("Error: This script must be run inside a git repository.", file=sys.stderr)
        return False

    # Check gh
    gh_check = run_command(["gh", "--version"], silent=True)
    if not gh_check:
        print("Error: GitHub CLI (gh) not found or not in PATH.", file=sys.stderr)
        print(
            "Please install GitHub CLI (https://cli.github.com/) and ensure 'gh' is available on your PATH.",
            file=sys.stderr,
        )
        return False

    gh_auth = run_command(["gh", "auth", "status"], silent=True)
    if not gh_auth:
        print("Error: GitHub CLI not authenticated. Please run 'gh auth login'.", file=sys.stderr)
        return False

    return True


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
    output = run_command(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", oid], silent=True)
    if output:
        return output.strip().splitlines()
    return []


def get_pr_number_from_subject(subject):
    """Extract PR number from a commit subject like 'Some title (#12345)'."""
    match = re.search(r"\(#(\d+)\)$", subject.strip())
    if match:
        return match.group(1)
    return None


def get_existing_pr_numbers(branch):
    """Get the set of PR numbers already present in the target branch."""
    output = run_command(["git", "log", branch, "--oneline", "-n", "500"], silent=True)
    if not output:
        return set()
    pr_numbers = set()
    for line in output.strip().splitlines():
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        subject = parts[1]
        pr_num = get_pr_number_from_subject(subject)
        if pr_num:
            pr_numbers.add(int(pr_num))
    return pr_numbers


def check_missing_dependencies(prs, branch):
    """Check for potential missing dependencies (conflicts)."""
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

        files = get_changed_files(oid)
        if not files:
            continue

        # For each file, find commits that modified it between the target branch and the cherry-picked commit.
        # Deduplicate warnings: group affected files by missing commit.
        # missing_commits maps: missing_commit_oid -> (title, [list of affected files])
        missing_commits = defaultdict(lambda: ("", []))

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
                    existing_title, existing_files = missing_commits[c]
                    if not existing_title:
                        existing_title = title
                    existing_files.append(filepath)
                    missing_commits[c] = (existing_title, existing_files)

        # Print deduplicated warnings
        for missing_oid, (title, affected_files) in missing_commits.items():
            files_str = ", ".join(affected_files)
            print(
                f"WARNING: PR #{number} ({oid}) modifies files that were also changed by commit {missing_oid} ({title}), "
                f"which is not in the cherry-pick list. This may indicate missing related changes. Affected files: {files_str}"
            )


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
    parser.add_argument("--limit", type=int, default=200, help="Wait limitation for PR fetching (default: 200)")
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
    existing_prs = get_existing_pr_numbers(args.branch)
    if existing_prs:
        print(f"Found {len(existing_prs)} PRs already in branch '{args.branch}'.")

    # Determine output format based on file extension
    is_shell = args.output.endswith(".sh")

    # 2. Write Output Script
    commit_count = 0
    skipped_count = 0
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

        for pr in prs:
            number = pr["number"]
            title = pr["title"]
            safe_title = title.replace("\n", " ")

            if not pr.get("mergeCommit"):
                print(f"Warning: PR #{number} has no merge commit OID. Skipping.", file=sys.stderr)
                continue

            if number in existing_prs:
                print(f"Skipping PR #{number} (already in branch '{args.branch}'): {safe_title}")
                skipped_count += 1
                continue

            oid = pr["mergeCommit"]["oid"]
            comment = "#" if is_shell else "rem"
            f.write(f"{comment} PR {number}: {safe_title}\n")
            f.write(f"git cherry-pick {oid}\n\n")
            commit_count += 1

    print(f"Generated {args.output} with {commit_count} commits ({skipped_count} skipped, already in branch).")

    # 3. Write PR Description Markdown (table format)
    md_output = "cherry_pick_pr_description.md"
    with open(md_output, "w", encoding="utf-8") as f:
        f.write("This cherry-picks the following commits for the release:\n\n")
        f.write("| Commit ID | PR Number | Commit Title |\n")
        f.write("|-----------|-----------|-------------|\n")
        for pr in prs:
            if not pr.get("mergeCommit"):
                continue
            number = pr["number"]
            if number in existing_prs:
                continue
            title = pr["title"].replace("\n", " ")
            oid = pr["mergeCommit"]["oid"]
            short_oid = oid[:10]
            f.write(f"| {short_oid} | #{number} | {title} |\n")

    print(f"Generated {md_output} with {commit_count} commits.")

    # 4. Dependency Check
    check_missing_dependencies(prs, args.branch)


if __name__ == "__main__":
    main()
