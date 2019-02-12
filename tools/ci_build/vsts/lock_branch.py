#!/usr/bin/env python3

import argparse
import os

import requests

def parse_args():
    parser = argparse.ArgumentParser(description="Locks or unlocks the specified branch.")

    parser.add_argument("--branch-name", required=True,
                        help="The name of the branch.")

    parser.add_argument("--lock", action="store_const", const=True, default=True,
                        help="Lock the branch (default).")
    parser.add_argument("--unlock", dest="lock", action="store_const", const=False,
                        help="Unlock the branch.")

    parser.add_argument("--token-env", required=True,
                        help="The environment variable containing the access token value.")

    return parser.parse_args()

def az_dev_ops_request(method, url, token, params=dict(), headers=dict(), **kwargs):
    """Makes an Azure Dev Ops API request.

    The signature is similar to requests.request(), with an additional token parameter.

    Args:
        token - the access token
        see requests.request() for other parameters

    Returns:
        A requests.Response instance.
    """
    base_headers = {"Authorization": "Bearer {}".format(token)}
    base_params = {"api-version": "5.0"}

    request_headers = dict(base_headers)
    base_headers.update(headers)

    request_params = dict(base_params)
    request_params.update(params)

    response = requests.request(method, url, params=request_params, headers=request_headers, **kwargs)

    return response

def check_response(response, valid_status_code_list):
    """Checks if the response status code is in the expected values, raises an exception if not."""
    if response.status_code not in valid_status_code_list:
        raise RuntimeError(
            "Unexpected response status code ({})! Response text: {}".format(
                response.status_code, response.text))

def main():
    account = "aiinfra"
    project = "lotus"
    repo_name = "onnxruntime"
    az_dev_ops_base_url = "https://dev.azure.com/{}/{}".format(account, project)

    args = parse_args()
    token = os.environ[args.token_env]

    response = az_dev_ops_request(
        method="PATCH",
        url="{}/_apis/git/repositories/{}/refs".format(az_dev_ops_base_url, repo_name),
        token=token,
        params={
            "projectId": project,
            "filter": "heads/{}".format(args.branch_name),
        },
        json={
            "isLocked": args.lock,
        })

    check_response(response, [200])

if __name__ == "__main__":
    import sys
    sys.exit(main())
