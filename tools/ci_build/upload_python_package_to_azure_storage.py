#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import os
import subprocess
import warnings

log = logging.getLogger("Build")


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


def run_subprocess(args, cwd=None):
    log.warning(f"Running subprocess in '{cwd or os.getcwd()}'\n{args}")
    return subprocess.run(args, cwd=cwd, check=True)


def upload_whl(python_wheel_path, final_storage=False):
    storage_account_name = "onnxruntimepackages" if final_storage else "onnxruntimepackagesint"
    blob_name = os.path.basename(python_wheel_path)
    run_subprocess(["azcopy", "cp", python_wheel_path, f"https://{storage_account_name}.blob.core.windows.net/$web/"])

    nightly_build, local_version = parse_nightly_and_local_version_from_whl_name(blob_name)
    if local_version:
        html_blob_name = f"onnxruntime_{nightly_build}_{local_version}.html"
    else:
        html_blob_name = f"onnxruntime_{nightly_build}.html"

    download_path_to_html = f"./onnxruntime_{nightly_build}.html"

    run_subprocess(
        [
            "azcopy",
            "cp",
            f"https://{storage_account_name}.blob.core.windows.net/$web/" + html_blob_name,
            download_path_to_html,
        ]
    )

    blob_name_plus_replaced = blob_name.replace("+", "%2B")
    with open(download_path_to_html) as f:
        lines = f.read().splitlines()

    new_line = f'<a href="{blob_name_plus_replaced}">{blob_name_plus_replaced}</a><br>'
    if new_line not in lines:
        lines.append(new_line)
        lines.sort()

        with open(download_path_to_html, "w") as f:
            for item in lines:
                f.write("%s\n" % item)
    else:
        warnings.warn(f"'{new_line}' exists in {download_path_to_html}. The html file is not updated.")
    run_subprocess(
        [
            "azcopy",
            "cp",
            download_path_to_html,
            f"https://{storage_account_name}.blob.core.windows.net/$web/" + html_blob_name,
            "--content-type",
            "text/html",
            "--overwrite",
            "true",
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload python whl to azure storage.")

    parser.add_argument("--python_wheel_path", type=str, help="path to python wheel")
    parser.add_argument("--final_storage", action="store_true", help="upload to final storage")

    args = parser.parse_args()

    upload_whl(args.python_wheel_path, args.final_storage)
