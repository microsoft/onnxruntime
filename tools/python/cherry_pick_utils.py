# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import re
import subprocess
import sys


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

    # gh auth status outputs to stderr, so run_command returns empty stdout even on success.
    # Use subprocess directly to check the return code.
    try:
        auth_result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, check=False)
        if auth_result.returncode != 0:
            print("Error: GitHub CLI not authenticated. Please run 'gh auth login'.", file=sys.stderr)
            return False
    except FileNotFoundError:
        print("Error: GitHub CLI (gh) not found.", file=sys.stderr)
        return False

    return True


def get_pr_number_from_subject(subject):
    """Extract PR number from a commit subject like 'Some title (#12345)'."""
    match = re.search(r"\(#(\d+)\)$", subject.strip())
    if match:
        return match.group(1)
    return None


def extract_pr_numbers(text, strict=False):
    if not text:
        return []

    if strict:
        # Strict mode: Only look for (#123) with closing paren, full onnxruntime URLs,
        # or PR numbers in markdown table cells (| #123 |), or standalone #123 with clear boundaries.
        # This avoids noise from version numbers or external repo PRs
        # And it avoids matching truncated headlines like (#25... as PR #25
        patterns = [
            r"\(#(\d+)\)",  # (#123)
            r"microsoft/onnxruntime/pull/(\d+)",
            r"(?:^|\s|-)#(\d+)(?:\s|$)",  # #123 at start, or preceded by space/dash, and followed by space or end
            r"\|\s*#(\d+)\s*\|",  # | #123 | (markdown table cell)
        ]

        results = []
        for p in patterns:
            results.extend(re.findall(p, text))
        return [int(x) for x in set(results)]

    # Matches patterns like #123 or https://github.com/microsoft/onnxruntime/pull/123
    # Also handles ( #123) or similar in titles
    prs = re.findall(r"(?:#|/pull/)(\d+)", text)
    return [int(x) for x in set(prs)]
