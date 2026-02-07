# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Compile Contributors Script
---------------------------
Description:
    This script compiles contributor information by comparing two git branches/commits.
    It identifies Pull Requests, handles cherry-picked commits (including "Cherry-pick round" meta-PRs),
    and consolidates author identities (names vs usernames).

Usage:
    python compile_contributors.py [--base <base_branch>] [--target <target_branch>] [--dir <output_dir>]

Example:
    python compile_contributors.py --base origin/rel-1.23.2 --target origin/rel-1.24.1 --dir rel-1.24.1_report

Outputs:
    - detail.csv: Detailed breakdown of PRs, authors, and commit links.
    - logs.txt: Processing logs and summary (professional humans-only contributor list for release notes).

Requirements:
    - Python 3.7+
    - GitHub CLI (gh) logged in.
"""

import argparse
import csv
import datetime
import json
import os
import re
import subprocess


def log_event(message, log_file=None):
    """Log a message to the console and an optional log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(message)  # Clean print for console UI
    if log_file:
        log_file.write(full_message + "\n")


def run_command(command_list, cwd=".", silent=False):
    """Run a command using a list of arguments for security (no shell=True)."""
    result = subprocess.run(command_list, check=False, capture_output=True, text=True, cwd=cwd, encoding="utf-8")
    if result.returncode != 0:
        if not silent:
            log_str = " ".join(command_list)
            print(f"Error running command: {log_str}")
            if result.stderr:
                print(f"Stderr: {result.stderr.strip()}")
        return None
    return result.stdout


def check_preflight():
    """Verify gh CLI and git repository early."""
    # Check git
    git_check = run_command(["git", "rev-parse", "--is-inside-work-tree"], silent=True)
    if not git_check:
        print("Error: This script must be run inside a git repository.")
        return False

    # Check gh
    gh_check = run_command(["gh", "--version"], silent=True)
    if not gh_check:
        print("Error: GitHub CLI (gh) not found or not in PATH.")
        return False

    gh_auth = run_command(["gh", "auth", "status"], silent=True)
    if not gh_auth:
        print("Error: GitHub CLI not authenticated. Please run 'gh auth login'.")
        return False

    return True


# Constants
PR_CACHE = {}  # Cache for PR details to speed up multiple rounds referencing same PRs
NAME_TO_LOGIN = {}  # Map full names to GitHub logins for consolidation
VERIFIED_LOGINS = set()  # Track IDs known to be valid GitHub logins (vs free-form names)

# Bots to exclude from contributor lists
BOT_NAMES = {
    "Copilot",
    "dependabot[bot]",
    "app/dependabot",
    "github-actions[bot]",
    "app/copilot-swe-agent",
    "CI Bot",
    "github-advanced-security[bot]",
    "GitHub Actions",
    "dependabot",
    "github-actions",
    "Gemini",
    "CI",
}


def is_bot(name):
    if not name:
        return True
    name_clean = name.strip().lstrip("@")
    # Known bots and patterns
    if name_clean in BOT_NAMES:
        return True
    if "[bot]" in name_clean.lower():
        return True
    if name_clean.lower().startswith("app/"):
        return True
    return False


def is_invalid(name):
    if not name:
        return True
    # If it's a bot, it's considered a valid identity for the CSV
    if is_bot(name):
        return False

    name_clean = name.strip().lstrip("@")
    # Paths, brackets, and code extensions
    if "/" in name_clean or "\\" in name_clean or "[" in name_clean or "]" in name_clean:
        return True
    if any(name_clean.lower().endswith(ext) for ext in [".cmake", ".py", ".h", ".cc", ".cpp", ".txt", ".md"]):
        return True
    return False


def get_pr_number(subject):
    match = re.search(r"\(#(\d+)\)$", subject.strip())
    if match:
        return match.group(1)
    return None


def get_pr_details(pr_number):
    if pr_number in PR_CACHE:
        return PR_CACHE[pr_number]

    # Try as a PR first - fetch author and commits to get all contributors
    output = run_command(["gh", "pr", "view", pr_number, "--json", "number,title,author,body,commits"], silent=True)
    if output:
        details = json.loads(output)
        PR_CACHE[pr_number] = details
        return details

    PR_CACHE[pr_number] = None
    return None


def extract_authors_from_pr(details):
    authors = set()
    if not details:
        return authors

    # Add main PR author
    pr_login = None
    if details.get("author"):
        pr_login = details["author"]["login"]
        authors.add(pr_login)
        VERIFIED_LOGINS.add(pr_login.lower())

    # Add authors from all commits in the PR
    if "commits" in details:
        for commit in details["commits"]:
            for author_info in commit.get("authors", []):
                login = author_info.get("login")
                name = author_info.get("name")
                if login:
                    authors.add(login)
                    VERIFIED_LOGINS.add(login.lower())
                    if name:
                        NAME_TO_LOGIN[name] = login
                elif name:
                    # HEURISTIC: If there is no login but there is a name,
                    # and we have a PR author login, associate this name with the PR author.
                    # This handles squash-merged PRs where Git name != GitHub handle.
                    if pr_login:
                        authors.add(pr_login)
                        NAME_TO_LOGIN[name] = pr_login
                    else:
                        authors.add(name)

    return authors


def extract_authors_from_commit(commit_id):
    authors = set()
    # Format: AuthorName \n Body
    info = run_command(["git", "show", "-s", "--format=%an%n%B", commit_id], silent=True)
    if not info:
        return authors

    lines = info.strip().splitlines()
    if lines:
        authors.add(lines[0])  # Main author name

    # Look for Co-authored-by trailers
    for line in lines:
        if "co-authored-by:" in line.lower():
            # Pattern: Co-authored-by: Name <email> or Co-authored-by: login <email>
            match = re.search(r"co-authored-by:\s*(.*?)\s*<", line, re.IGNORECASE)
            if match:
                authors.add(match.group(1).strip())

    return authors


def extract_pr_numbers(text, strict=False):
    if not text:
        return []

    if strict:
        # Strict mode: Only look for (#123) with closing paren or full onnxruntime URLs
        # This avoids noise from version numbers or external repo PRs
        # And it avoids matching truncated headlines like (#25... as PR #25
        patterns = [
            r"\(#(\d+)\)",  # (#123)
            r"microsoft/onnxruntime/pull/(\d+)",
        ]
        results = []
        for p in patterns:
            results.extend(re.findall(p, text))
        return [int(x) for x in set(results)]

    # Matches patterns like #123 or https://github.com/microsoft/onnxruntime/pull/123
    # Also handles ( #123) or similar in titles
    prs = re.findall(r"(?:#|/pull/)(\d+)", text)
    return [int(x) for x in set(prs)]


def get_prs_from_log(log_output, prs_base=None, log_file=None, scan_depth=100):
    if not log_output:
        return {}

    all_prs = {}  # pr_number -> {title, authors, original_pr, cherry_pick_commit}
    lines = log_output.splitlines()
    total_commits = len(lines)
    commit_count = 0

    log_event(f"Processing {total_commits} commits...", log_file)

    for line in lines:
        commit_count += 1
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        commit_id = parts[0]
        subject = parts[1]

        # Concise progress indicator
        pr_num_str = get_pr_number(subject)
        display_id = f"PR #{pr_num_str}" if pr_num_str else f"commit {commit_id}"
        log_event(f"[{commit_count}/{total_commits}] Processing {display_id}...", log_file)

        details = None
        if pr_num_str:
            if prs_base and pr_num_str in prs_base:
                log_event(f"  - PR #{pr_num_str} already in base branch, skipping.", log_file)
                continue
            details = get_pr_details(pr_num_str)

        if details:
            # Check if it's a cherry-pick round PR
            is_meta_pr = (
                "cherry pick" in subject.lower() or "cherry-pick" in subject.lower() or "cherrypick" in subject.lower()
            )

            if is_meta_pr and commit_count < scan_depth:
                log_event(f"  - Meta-PR detected, expanding: {details['title']}", log_file)
                # Collect Original PRs from Title, Body, and Commits
                all_extracted_nums = []
                # Use strict extraction even for titles to avoid matching issues like #26985
                all_extracted_nums.extend(extract_pr_numbers(details["title"], strict=True))
                all_extracted_nums.extend(extract_pr_numbers(details["body"], strict=True))

                # Reuse commits already fetched in get_pr_details to avoid an extra gh CLI call
                for commit in details.get("commits", []):
                    all_extracted_nums.extend(extract_pr_numbers(commit.get("messageHeadline", ""), strict=True))
                    all_extracted_nums.extend(extract_pr_numbers(commit.get("messageBody", ""), strict=True))

                # Filter and Normalize
                current_pr_int = int(pr_num_str)
                valid_pr_ints = []
                for op_num in set(all_extracted_nums):
                    if op_num == current_pr_int:
                        continue
                    valid_pr_ints.append(op_num)

                # Sorting results numerically (100 > 99)
                original_pr_ints = sorted(valid_pr_ints)
                log_event(f"  - Extracted sub-PR candidates: {original_pr_ints}", log_file)

                if original_pr_ints:
                    log_event(f"  -> Found {len(original_pr_ints)} sub-PRs for expansion.", log_file)
                    for op_num in original_pr_ints:
                        op_num_str = str(op_num)
                        if prs_base and op_num_str in prs_base:
                            log_event(f"    - Sub-PR #{op_num_str} already in base branch, skipping.", log_file)
                            continue

                        op_details = get_pr_details(op_num_str)
                        if op_details:
                            log_event(f"    - Added Sub-PR #{op_num_str}: {op_details['title']}", log_file)
                            all_prs[op_num_str] = {
                                "title": op_details["title"],
                                "authors": list(extract_authors_from_pr(op_details)),
                                "cherry_pick_commit": commit_id,
                                "cherry_pick_pr": pr_num_str,
                            }
                        else:
                            # If we can't resolve this number as a PR (e.g., issue reference or inaccessible/deleted PR),
                            # do not invent new authors, but still attribute it to the known meta-PR to avoid losing credit.
                            log_event(
                                f"    - Warning: Unable to resolve PR #{op_num_str} via GitHub CLI; attributing via meta-PR #{pr_num_str}.",
                                log_file,
                            )
                            if op_num_str not in all_prs:
                                fallback_title = (
                                    f"Unresolved sub-PR #{op_num_str} (attributed via meta-PR #{pr_num_str})"
                                )
                                all_prs[op_num_str] = {
                                    "title": fallback_title,
                                    "authors": list(extract_authors_from_pr(details)),
                                    "cherry_pick_commit": commit_id,
                                    "cherry_pick_pr": pr_num_str,
                                }
                else:
                    log_event("  - No sub-PRs found, treating meta-PR as a normal PR.", log_file)
                    all_prs[pr_num_str] = {
                        "title": details["title"],
                        "authors": list(extract_authors_from_pr(details)),
                        "cherry_pick_commit": commit_id,
                        "cherry_pick_pr": None,
                    }
            else:
                log_event(f"  - Added PR #{pr_num_str}: {details['title']}", log_file)
                all_prs[pr_num_str] = {
                    "title": details["title"],
                    "authors": list(extract_authors_from_pr(details)),
                    "cherry_pick_commit": commit_id,
                    "cherry_pick_pr": None,
                }
        else:
            # Not a PR OR PR detail fetch failed (e.g. it was an issue or deleted PR)
            # Use git commit author as the reliable fallback
            if pr_num_str:
                log_event(
                    f"  - PR #{pr_num_str} lookup failed (possibly issue or deleted). Falling back to commit author.",
                    log_file,
                )
            authors = extract_authors_from_commit(commit_id)
            if authors:
                log_event(f"  - Added commit {commit_id} with authors: {list(authors)}", log_file)
                all_prs[f"commit_{commit_id}"] = {
                    "title": subject,
                    "authors": list(authors),
                    "cherry_pick_commit": commit_id,
                    "cherry_pick_pr": None,
                }

    return all_prs


def main():
    parser = argparse.ArgumentParser(description="Compile contributor list from Git log comparison.")
    parser.add_argument("--base", default="origin/rel-1.23.2", help="Base branch/commit to compare from")
    parser.add_argument("--target", default="origin/rel-1.24.1", help="Target branch/commit to compare to")
    parser.add_argument("--dir", default="contributors", help="Output directory for reports and logs")
    parser.add_argument("--scan-depth", type=int, default=200, help="Depth to scan base/meta-PRs for deduplication")
    args = parser.parse_args()

    # Early validation
    if not check_preflight():
        return

    branch_base = args.base
    branch_target = args.target
    output_dir = args.dir
    scan_depth = args.scan_depth

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logs_path = os.path.join(output_dir, "logs.txt")
    with open(logs_path, "w", encoding="utf-8") as log_file:
        log_event(f"Starting comparison: {branch_base} -> {branch_target}", log_file)

        # 1. Fetch base branch PRs (scan depth controlled by scan_depth)
        log_event(f"Fetching base branch history for {branch_base} (last {scan_depth})...", log_file)
        log_base = run_command(["git", "log", branch_base, "-n", str(scan_depth), "--oneline"])
        if log_base is None:
            log_event(
                f"Error: Could not fetch history for base ref '{branch_base}'. Please check if the ref exists.",
                log_file,
            )
            return

        prs_base_dict = get_prs_from_log(log_base, prs_base=None, log_file=log_file, scan_depth=scan_depth)
        prs_base = set(prs_base_dict.keys())

        # 2. Fetch target branch PRs (only those not in base)
        log_event(f"Fetching target branch history: {branch_base}..{branch_target}...", log_file)
        # Using A..B syntax for git log
        log_target = run_command(["git", "log", f"{branch_base}..{branch_target}", "--oneline"])
        if log_target is None:
            log_event(
                f"Error: Could not fetch history for range '{branch_base}..{branch_target}'. Please check if the refs exist.",
                log_file,
            )
            return

        prs_target = get_prs_from_log(log_target, prs_base=prs_base, log_file=log_file, scan_depth=scan_depth)

        # All PRs in target but not in base (deduplicated by key)
        new_pr_keys = set(prs_target.keys())

        contributors = {}  # username -> count
        details = []

        # Use str directly as key for sorting
        for key in sorted(new_pr_keys, key=str):
            info = prs_target[key]
            authors = info.get("authors", [])

            # Count each author separately
            for author in authors:
                contributors[author] = contributors.get(author, 0) + 1

            details.append(
                {
                    "original_pr": key,
                    "title": info["title"],
                    "authors": "; ".join(authors),
                    "target_commit": info["cherry_pick_commit"],
                    "cherry_pick_pr": info["cherry_pick_pr"],
                }
            )

        # Consolidation Pass
        consolidated_contributors = {}  # login_lower -> count
        display_names = {}  # login_lower -> original_casing
        raw_contributors = {}  # login_lower -> count

        for contributor, count in contributors.items():
            # Map to final identity
            final_author = NAME_TO_LOGIN.get(contributor, contributor)
            author_lower = final_author.lower()

            if author_lower not in display_names:
                display_names[author_lower] = final_author

            raw_contributors[author_lower] = raw_contributors.get(author_lower, 0) + count

            # Human-only for summary: Filter bots AND invalid/path strings
            if not is_bot(contributor) and not is_invalid(contributor):
                if not is_bot(final_author) and not is_invalid(final_author):
                    consolidated_contributors[author_lower] = consolidated_contributors.get(author_lower, 0) + count

        # Sort human contributors by count descending, then alphabetically by identity for determinism
        sorted_contributors = sorted(consolidated_contributors.items(), key=lambda x: (-x[1], x[0]))

        log_event("\n--- Summary ---", log_file)
        # Prefix only identified github logins and format as markdown links
        output_users = []
        for login_lower, _login in sorted_contributors:
            u = display_names[login_lower]
            if login_lower in VERIFIED_LOGINS:
                output_users.append(f"[@{u}](https://github.com/{u})")
            else:
                output_users.append(u)

        # Summary text as a single line for best copy-paste behavior in GitHub
        summary_text = ", ".join(output_users)
        log_event(summary_text, log_file)

        # Write details to CSV in the output directory
        csv_path = os.path.join(output_dir, "detail.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["original_pr", "authors", "title", "target_commit", "cherry_pick_pr"]
            )
            writer.writeheader()
            for row in details:
                # Consolidate authors in CSV as well
                authors_list = [a.strip() for a in row["authors"].split(";")]
                consolidated_authors = []
                for a in authors_list:
                    # In CSV, we KEEP bots but filter out truly invalid entries (paths)
                    final_a = NAME_TO_LOGIN.get(a, a)
                    # Normalize for uniqueness check
                    if not is_invalid(final_a):
                        consolidated_authors.append(final_a)

                # Deduplicate while preserving case-insensitive uniqueness
                unique_authors = {}
                for a in consolidated_authors:
                    unique_authors[a.lower()] = a
                row["authors"] = "; ".join(sorted(unique_authors.values(), key=lambda x: x.lower()))
                writer.writerow(row)

        log_event(
            f"\nDetailed information written to {csv_path}. Total human contributors: {len(consolidated_contributors)}",
            log_file,
        )


if __name__ == "__main__":
    main()
