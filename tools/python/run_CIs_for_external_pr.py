#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import subprocess
import sys
import typing

parser = argparse.ArgumentParser(
    os.path.basename(__file__),
    description="""Trigger CIs running for the specified pull request.

    Requires the GitHub CLI to be installed. 
    See https://github.com/cli/cli#installation for details.
    After installation you will also need to setup an auth token to access the ONNX Runtime repository by running
    `gh auth login`. Easiest is to run that from a directory in your local copy of the repo.
    """
)
parser.add_argument("--pr", required=True, help="Specify the pull request ID.")
args = parser.parse_args()

# Current pipelines. These change semi-frequently and may need updating.
windows_pipelines = (
    "Windows CPU CI Pipeline,Windows GPU CI Pipeline,Windows GPU TensorRT CI Pipeline,ONNX Runtime Web CI Pipeline"
)
checks_pipelines = "onnxruntime-python-checks-ci-pipeline,onnxruntime-binary-size-checks-ci-pipeline"
linux_pipelines = (
    "Linux CPU CI Pipeline,Linux CPU Minimal Build E2E CI Pipeline,Linux GPU CI Pipeline,"
    "Linux GPU TensorRT CI Pipeline,Linux Nuphar CI Pipeline,Linux OpenVINO CI Pipeline"
)
mac_pipelines = "MacOS CI Pipeline"
training_pipelines = (
    "orttraining-amd-gpu-ci-pipeline,orttraining-linux-ci-pipeline,orttraining-linux-gpu-ci-pipeline,"
    "orttraining-ortmodule-distributed"
)


def run_gh_pr_command(command: typing.List[str]):
    try:
        return subprocess.run(["gh", "pr"] + command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as cpe:
        print(cpe)
        print(cpe.stderr)
        sys.exit(-1)


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

print("Adding azp run commands")

run_gh_pr_command(["comment", pr_id, "--body", f"/azp run {windows_pipelines},{checks_pipelines}"])
run_gh_pr_command(["comment", pr_id, "--body", f"/azp run {linux_pipelines},{mac_pipelines}"])
run_gh_pr_command(["comment", pr_id, "--body", f"/azp run {training_pipelines}"])

print(f"Done. Check status at https://github.com/microsoft/onnxruntime/pull/{pr_id}/checks")
