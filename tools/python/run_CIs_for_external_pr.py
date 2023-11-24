#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import subprocess
import sys
import typing


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Trigger CIs running for the specified pull request.

        Requires the GitHub CLI to be installed. See https://github.com/cli/cli#installation for details.
        After installation you will also need to setup an auth token to access the ONNX Runtime repository by running
        `gh auth login`. Easiest is to run that from a directory in your local copy of the repo.
        """,
    )
    parser.add_argument("pr", help="Specify the pull request ID.")
    args = parser.parse_args()
    return args


def run_gh_pr_command(command: typing.List[str], check=True):
    try:
        return subprocess.run(["gh", "pr", *command], capture_output=True, text=True, check=check)
    except subprocess.CalledProcessError as cpe:
        print(cpe)
        print(cpe.stderr)
        sys.exit(-1)


def main():
    args = parse_args()
    pr_id = args.pr

    # validate PR
    gh_out = run_gh_pr_command(["view", pr_id])
    info = gh_out.stdout.split("\n")
    for line in info:
        pieces = line.split("\t")
        if len(pieces) != 2:
            continue

        if pieces[0] == "state:":
            if pieces[1] != "OPEN":
                print(f"PR {pr_id} is not OPEN. Currently in state {pieces[1]}.")
                sys.exit(-1)

    print("Check passed pipelines")
    gh_out = run_gh_pr_command(["checks", pr_id, "--required"], check=False)
    # output format is a tab separated list of columns:
    # (pipeline name) "\t" (status) "\t" (ran time) "\t" (url)
    checked_pipelines = [
        columns[0]
        for columns in (line.strip().split("\t") for line in gh_out.stdout.split("\n"))
        if len(columns) == 4 and columns[1] == "pass"
    ]

    print("Adding azp run commands")

    # Current pipelines. These change semi-frequently and may need updating.
    #
    # Note: there is no easy way to get the list for azp "required" pipelines before they starts.
    #       we need to maintain this list manually.
    #
    pipelines = [
        # windows
        "Windows ARM64 QNN CI Pipeline",
        "Windows x64 QNN CI Pipeline",
        "Windows CPU CI Pipeline",
        "Windows GPU CI Pipeline",
        "Windows GPU TensorRT CI Pipeline",
        "ONNX Runtime Web CI Pipeline",
        # linux
        "Linux CPU CI Pipeline",
        "Linux CPU Minimal Build E2E CI Pipeline",
        "Linux GPU CI Pipeline",
        "Linux GPU TensorRT CI Pipeline",
        "Linux OpenVINO CI Pipeline",
        "Linux QNN CI Pipeline",
        # mac
        "MacOS CI Pipeline",
        # training
        "orttraining-amd-gpu-ci-pipeline",
        "orttraining-linux-ci-pipeline",
        "orttraining-linux-gpu-ci-pipeline",
        "orttraining-ortmodule-distributed",
        # checks
        "onnxruntime-python-checks-ci-pipeline",
        "onnxruntime-binary-size-checks-ci-pipeline",
        # not currently required, but running ensures we're hitting all mobile platforms
        "Android CI Pipeline",
        "iOS CI Pipeline",
        "ONNX Runtime React Native CI Pipeline",
    ]

    # remove pipelines that have already run successfully
    pipelines = [p for p in pipelines if p not in checked_pipelines]

    # azp run is limited to 10 pipelines at a time
    max_pipelines_per_comment = 10
    start = 0
    num_pipelines = len(pipelines)

    while start < num_pipelines:
        end = start + max_pipelines_per_comment
        if end > num_pipelines:
            end = num_pipelines

        run_gh_pr_command(["comment", pr_id, "--body", f"/azp run {str.join(',', pipelines[start:end])}"])

        start += max_pipelines_per_comment

    print(f"Done. Check status at https://github.com/microsoft/onnxruntime/pull/{pr_id}/checks")


if __name__ == "__main__":
    main()
