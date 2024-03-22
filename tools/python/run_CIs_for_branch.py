#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import subprocess
import sys
import typing

from run_CIs_for_external_pr import get_pipeline_names
from util.platform_helpers import is_windows


class DefaultArgsRawHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        formatter_class=DefaultArgsRawHelpFormatter,
        description="""Run the CIs used to validate PRs for the specified branch.

        If not specified, the branch will be inferred (if possible) by running `git branch --show-current`.

        If specified, the `--include` filter is applied first, followed by any `--exclude` filter.
        `--include` and `--exclude` can be specified multiple times to accumulate values to include/exclude.

        Requires the Azure CLI with DevOps extension to be installed.
          Azure CLI: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
          DevOps extension: https://github.com/Azure/azure-devops-cli-extension

        Configuration:
          Login:`az login`
          Configure ORT repo as default:
            `az devops configure --defaults organization=https://dev.azure.com/onnxruntime project=onnxruntime`

        Example usage:
          List all CIs
            `python run_CIs_for_branch.py --dry-run my/BranchName`
          Run all CIs
            `python run_CIs_for_branch.py my/BranchName`
          Run only Linux CIs
            `python run_CIs_for_branch.py --include linux my/BranchName`
          Exclude training CIs
            `python run_CIs_for_branch.py --exclude training my/BranchName`
          Run non-training Linux CIs
            `python run_CIs_for_branch.py --include linux --exclude training my/BranchName`
        """,
    )

    current_branch = None
    get_branch_result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, check=False)
    if get_branch_result.returncode == 0:
        current_branch = get_branch_result.stdout.strip()

    parser.add_argument(
        "-i", "--include", action="append", type=str, help="Include CIs that match this string. Case insensitive."
    )
    parser.add_argument(
        "-e", "--exclude", action="append", type=str, help="Exclude CIs that match this string. Case insensitive."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected CIs but do not run them.")
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


def _run_az_pipelines_command(command: typing.List[str]):
    try:
        az = "az.cmd" if is_windows() else "az"
        az_output = subprocess.run([az, "pipelines", *command], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as cpe:
        print(cpe)
        print(cpe.stderr)
        sys.exit(-1)

    return az_output


def main():
    args = _parse_args()
    branch = args.branch

    # To debug available pipelines:
    # az_out = az_pipelines = _run_az_pipelines_command(["list"])
    # pipeline_info = json.loads(az_out.stdout)
    # print(pipeline_info)

    pipelines = get_pipeline_names()
    pipelines_to_run = []
    if args.include:
        values = [i.lower().strip() for i in args.include]
        for p in pipelines:
            include = False
            for value in values:
                if value in p.lower():
                    include = True
                    break

            if include:
                print(f"Including {p}")
                pipelines_to_run.append(p)
    else:
        pipelines_to_run = pipelines

    if args.exclude:
        values = [e.lower().strip() for e in args.exclude]
        cur_pipelines = pipelines_to_run
        pipelines_to_run = []
        for p in cur_pipelines:
            exclude = False
            for value in values:
                if value in p.lower():
                    exclude = True
                    break

            if exclude:
                print(f"Excluding {p}")
            else:
                pipelines_to_run.append(p)

    print(f"Pipelines to run for {args.branch}:")
    for p in pipelines_to_run:
        print(f"\t{p}")

    if args.dry_run:
        sys.exit(0)

    for pipeline in pipelines_to_run:
        az_out = _run_az_pipelines_command(["run", "--branch", branch, "--name", pipeline])
        run_output = json.loads(az_out.stdout)
        if "id" in run_output:
            build_url = f"https://dev.azure.com/onnxruntime/onnxruntime/_build/results?buildId={run_output['id']}"
            print(f"{pipeline} build results: {build_url}&view=results")
        else:
            raise ValueError("Build id was not found in az output:\n" + run_output)


if __name__ == "__main__":
    main()
