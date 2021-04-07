#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
from azure.storage.blob import BlockBlobService


def upload_whl(python_wheel_path, account_name, account_key, container_name):
    block_blob_service = BlockBlobService(
        account_name=account_name,
        account_key=account_key
    )

    blob_name = os.path.basename(python_wheel_path)
    block_blob_service.create_blob_from_path(container_name, blob_name, python_wheel_path)

    html_blob_name = 'onnxruntime_nightly.html'
    download_path_to_html = "./onnxruntime_nightly.html"
    block_blob_service.get_blob_to_path(container_name, html_blob_name, download_path_to_html)

    blob_name_plus_replaced = blob_name.replace('+', '%2B')
    with open(download_path_to_html) as f:
        lines = f.read().splitlines()

    new_line = '<a href="{blobname}">{blobname}</a><br>'.format(blobname=blob_name_plus_replaced)
    lines.append(new_line)
    lines.sort()

    with open(download_path_to_html, 'w') as f:
        for item in lines:
            f.write("%s\n" % item)

    block_blob_service.create_blob_from_path(container_name, html_blob_name, download_path_to_html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload python whl to azure storage.")

    parser.add_argument("--python_wheel_path", type=str, help="path to python wheel")
    parser.add_argument("--account_name", type=str, help="account name")
    parser.add_argument("--account_key", type=str, help="account key")
    parser.add_argument("--container_name", type=str, help="container name")

    args = parser.parse_args()

    upload_whl(args.python_wheel_path, args.account_name, args.account_key, args.container_name)
