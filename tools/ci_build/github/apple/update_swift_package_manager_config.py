#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import hashlib
import os
import pathlib
from typing import Union


def calc_checksum(filename: pathlib.Path):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        while chunk := f.read(4096):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def checksum_remote_package(package_url: str):
    import tempfile
    from urllib.parse import urlparse
    from urllib.request import urlretrieve

    # tempfile.NamedTemporaryFile returns an open file, but we need a filename
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.basename(package_url)
        assert filename

        temp_file = os.path.join(tmp, filename)
        print(f"Downloading file from {package_url} to {temp_file}")
        urlretrieve(package_url, filename=temp_file)
        return calc_checksum(temp_file)


# find the below section in Package.swift and replace the values for url: and checksum:.
# If replacing with a local file, 'url:' -> 'file:' and we don't have a checksum
#
#   .binaryTarget(name: "onnxruntime",
#      url: "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-c-1.14.0.zip",
#      checksum: "c89cd106ff02eb3892243acd7c4f2bd8e68c2c94f2751b5e35f98722e10c042b"),
#
def update_swift_package(spm_config_path: pathlib.Path, ort_package_path: Union[pathlib.Path, str]):
    # ort_package_path must be a str for a url or Path for local file
    is_url = type(ort_package_path) is str
    if is_url:
        checksum = checksum_remote_package(ort_package_path)
    else:
        checksum = calc_checksum(ort_package_path)

    updated = False
    new_config = []
    with open(spm_config_path, "r") as config:
        while line := config.readline():
            if '.binaryTarget(name: "onnxruntime"' in line:
                # find and update the following 2 lines
                url_line = config.readline()
                checksum_line = config.readline()
                assert "url:" in url_line
                assert "checksum:" in checksum_line

                if is_url:
                    start_url_value = url_line.find('"')
                    new_url_line = url_line[: start_url_value + 1] + str(ort_package_path) + '",\n'
                else:
                    start_url = url_line.find("url:")
                    new_url_line = url_line[:start_url] + f'path: "{ort_package_path}"),\n'

                new_config.append(line)
                new_config.append(new_url_line)

                if is_url:
                    start_checksum_value = checksum_line.find('"')
                    new_checksum_line = checksum_line[: start_checksum_value + 1] + checksum + '"),\n'
                    new_config.append(new_checksum_line)

                updated = True
            else:
                new_config.append(line)

    assert updated

    # overwrite with new content
    with open(spm_config_path, "w") as new_config_f:
        for line in new_config:
            new_config_f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        f"{os.path.basename(__file__)}",
        description="Update the ORT binary used in the Package.swift config file.",
    )

    parser.add_argument(
        "--spm_config",
        type=pathlib.Path,
        required=True,
        help="Full path to Package.swift, the Swift Package Manager config.",
    )

    parser.add_argument(
        "--ort_package",
        type=str,
        required=True,
        help="ORT native iOS pod to use. Can be a url starting with 'http' or local file. "
        "The package name should look something like pod-archive-onnxruntime-c-<version info>.zip. "
        "If it's a local file it must have a path relative to the location of Package.swift. "
        "Copying it to the /swift directory is recommended.",
    )

    args = parser.parse_args()
    spm_config = args.spm_config.resolve(strict=True)

    if args.ort_package.startswith("http"):
        ort_package = args.ort_package
    else:
        # we need to leave this as a relative path to make SPM happy so don't call resolve
        ort_package = pathlib.Path(args.ort_package)

    update_swift_package(spm_config, ort_package)
