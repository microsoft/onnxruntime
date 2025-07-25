#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os
import subprocess

import yaml


def get_repo_root():
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("❌ Not inside a Git repository.")
    return result.stdout.strip()


def get_current_branch():
    result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def get_dispatchable_workflows(
    workflows_dir: str, include: list[str] | None = None, exclude: list[str] | None = None
) -> list[tuple[str, str]]:
    dispatchable = []
    for file in os.listdir(workflows_dir):
        if not file.endswith(".yml"):
            continue

        filepath = os.path.join(workflows_dir, file)
        with open(filepath, encoding="utf-8") as f:
            try:
                content = yaml.safe_load(f)
                triggers = content.get("on") or content.get(True)
                name = content.get("name")
                is_dispatchable = False
                if name and triggers:
                    if isinstance(triggers, dict) and "workflow_dispatch" in triggers:
                        is_dispatchable = True
                    elif isinstance(triggers, list) and "workflow_dispatch" in triggers:
                        is_dispatchable = True
                    elif isinstance(triggers, str) and triggers == "workflow_dispatch":
                        is_dispatchable = True

                    if is_dispatchable:
                        add = not include  # true unless there's an include filter
                        if include and any(inc.lower() in name.lower() for inc in include):
                            add = True

                        if exclude and any(exc.lower() in name.lower() for exc in exclude):
                            add = False

                        if add:
                            dispatchable.append((name, file))
            except yaml.YAMLError as e:
                print(f"⚠️ Failed to parse {file}: {e}")
    return dispatchable


def trigger_workflow(name: str, file_name: str, branch: str, dry_run: bool = False):
    command = ["gh", "workflow", "run", file_name, "--ref", branch]
    print(f"Workflow:{name}\n\t{' '.join(command)}")

    if dry_run:
        return

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        print(f"✅ Triggered {file_name} on branch {branch}")
    else:
        print(f"❌ Failed to trigger {file_name}:\n{result.stderr}")


class DefaultArgsRawHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        formatter_class=DefaultArgsRawHelpFormatter,
        description="""Run the GitHub workflows that have workflow_dispatch enabled for a branch.

        If specified, the `--include` filter is applied first, followed by any `--exclude` filter.
        `--include` and `--exclude` can be specified multiple times to accumulate values to include/exclude.

        Requires the GitHub CLI to be installed.

        Example usage:
          List all workflows that can be triggered.
            `python run_workflows_for_branch.py --dry-run [my/BranchName]`
          Run all workflows.
            `python run_workflows_for_branch.py [my/BranchName]`
          Run only Linux CIs
            `python run_workflows_for_branch.py --include linux [my/BranchName]`
          Exclude training CIs
            `python run_workflows_for_branch.py --exclude training [my/BranchName]`
          Run non-training Linux CIs
            `python run_workflows_for_branch.py --include linux --exclude training [my/BranchName]`
        """,
    )

    current_branch = get_current_branch()

    parser.add_argument(
        "-i", "--include", action="append", type=str, help="Include workflows that match this string. Case insensitive."
    )
    parser.add_argument(
        "-e", "--exclude", action="append", type=str, help="Exclude workflows that match this string. Case insensitive."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected workflows but do not run them.")
    parser.add_argument(
        "branch",
        type=str,
        nargs="?",
        default=current_branch,
        help="Specify the branch to run. Default is current branch if available.",
    )

    args = parser.parse_args()
    if not args.branch:
        raise ValueError("Branch was unable to be inferred and must be specified")

    return args


def main():
    args = _parse_args()

    repo_root = get_repo_root()
    workflows_dir = os.path.join(repo_root, ".github", "workflows")

    print(f"Branch: {args.branch}")

    workflows = get_dispatchable_workflows(workflows_dir, args.include, args.exclude)
    if not workflows:
        print("⚠️ No dispatchable workflows found.")
        return

    print(f"🔍 Found {len(workflows)} dispatchable workflows:")
    for wf in workflows:
        name = wf[0]
        file_name = wf[1]
        trigger_workflow(name, file_name, args.branch, args.dry_run)


if __name__ == "__main__":
    main()
