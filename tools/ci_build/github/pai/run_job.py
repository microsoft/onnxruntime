#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import re
import sys
import time

import requests


pai_base_url = "https://rr.openpai.org/rest-server"


def parse_args():
    parser = argparse.ArgumentParser(description="Runs a job on PAI.")
    parser.add_argument("job_yaml_file", help="The job YAML file.")
    parser.add_argument("job_name", help="The job name.")
    parser.add_argument("--user-env", required=True, help="Environment variable containing the user name.")
    parser.add_argument("--token-env", required=True, help="Environment variable containing the authorization token.")
    parser.add_argument("--yaml-sub-env", action="append", nargs=2,
                        help="YAML substitution key and environment variable containing the value.")

    return parser.parse_args()


def get_yaml_text_with_substitutions(job_yaml_file_path, substitutions):
    substitution_pattern = re.compile(r"@@(\w+)@@")

    def replace(match):
        if match[1] in substitutions:
            return substitutions[match[1]]

        print("Warning - no substitution was provided for '{}'.".format(match[0]))
        return match[0]

    with open(job_yaml_file_path, mode="r") as yaml_file:
        return re.sub(substitution_pattern, replace, yaml_file.read())


def submit_job(yaml, token):
    url = "{}/api/v2/jobs".format(pai_base_url)
    headers = {
        "Authorization": "Bearer {}".format(token),
        "Content-Type": "text/yaml",
    }

    response = requests.post(url=url, data=yaml, headers=headers)
    response.raise_for_status()


def wait_for_job(job_name, user, token):
    url = "{}/api/v2/jobs/{}~{}".format(pai_base_url, user, job_name)
    headers = {
        "Authorization": "Bearer {}".format(token),
    }

    while True:
        response = requests.get(url=url, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        job_status = response_json["jobStatus"]["state"]

        if job_status in ["WAITING", "RUNNING"]:
            time.sleep(30)
        elif job_status == "SUCCEEDED":
            break
        else:
            raise RuntimeError("Job failed. Status query response:\n{}".format(json.dumps(response_json, indent=2)))


def main():
    args = parse_args()

    substitutions = {
        "job_name": args.job_name,
    }
    if args.yaml_sub_env is not None:
        substitutions.update({kvp[0]: os.environ.get(kvp[1], "") for kvp in args.yaml_sub_env})

    yaml_with_substitutions = get_yaml_text_with_substitutions(args.job_yaml_file, substitutions)
    user = os.environ[args.user_env]
    token = os.environ[args.token_env]

    print("Submitting job {} ..".format(args.job_name))
    sys.stdout.flush()
    submit_job(yaml_with_substitutions, token)
    print('See https://rr.openpai.org/job-detail.html?username={}&jobName={}'.format(user, args.job_name))
    sys.stdout.flush()

    print('\nWarning: The following tests will be excluded:')
    with open('tools/ci_build/github/pai/pai-excluded-tests.txt', 'r') as fin:
        print(fin.read())
    print('')

    print("Waiting for job to complete ..")
    sys.stdout.flush()
    wait_for_job(args.job_name, user, token)


if __name__ == "__main__":
    main()
