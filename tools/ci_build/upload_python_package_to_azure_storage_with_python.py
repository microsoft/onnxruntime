#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import warnings

from azure.storage.blob import BlockBlobService, ContentSettings


def parse_nightly_and_local_version_from_whl_name(blob_name):
    night_build = "nightly" if blob_name.find(".dev") > 0 else "stable"

    start = blob_name.find("+")
    if start == -1:
        return night_build, None
    start = start + 1
    end = blob_name.find("-", start)
    if end == -1:
        return night_build, None
    return night_build, blob_name[start:end]


def upload_whl(python_wheel_path, account_name, account_key, container_name):
    block_blob_service = BlockBlobService(account_name=account_name, account_key=account_key)

    blob_name = os.path.basename(python_wheel_path)
    block_blob_service.create_blob_from_path(container_name, blob_name, python_wheel_path)

    nightly_build, local_version = parse_nightly_and_local_version_from_whl_name(blob_name)
    if local_version:
        html_blob_name = "onnxruntime_{}_{}.html".format(nightly_build, local_version)
    else:
        html_blob_name = "onnxruntime_{}.html".format(nightly_build)

    download_path_to_html = "./onnxruntime_{}.html".format(nightly_build)

    block_blob_service.get_blob_to_path(container_name, html_blob_name, download_path_to_html)

    blob_name_plus_replaced = blob_name.replace("+", "%2B")
    with open(download_path_to_html) as f:
        lines = f.read().splitlines()

    new_line = '<a href="{blobname}">{blobname}</a><br>'.format(blobname=blob_name_plus_replaced)
    if new_line not in lines:
        lines.append(new_line)
        lines.sort()

        with open(download_path_to_html, "w") as f:
            for item in lines:
                f.write("%s\n" % item)
    else:
        warnings.warn("'{}' exists in {}. The html file is not updated.".format(new_line, download_path_to_html))

    content_settings = ContentSettings(content_type="text/html")
    block_blob_service.create_blob_from_path(
        container_name, html_blob_name, download_path_to_html, content_settings=content_settings
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload python whl to azure storage.")

    parser.add_argument("--python_wheel_path", type=str, help="path to python wheel")
    parser.add_argument("--account_name", type=str, help="account name")
    parser.add_argument("--account_key", type=str, help="account key")
    parser.add_argument("--container_name", type=str, help="container name")

    # TODO: figure out a way to secure args.account_key to prevent later code changes
    # that may accidentally print out it to the console.
    args = parser.parse_args()

    upload_whl(args.python_wheel_path, args.account_name, args.account_key, args.container_name)
