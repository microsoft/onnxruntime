#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Set plugin EP package version variables for Azure Pipelines."""

import argparse
import datetime
import os
import re
import subprocess
import sys


class AzurePipelinesArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print(f"##vso[task.logissue type=error]{message}")
        self.exit(2)


def parse_arguments():
    parser = AzurePipelinesArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-version",
        choices=("release", "rc", "dev"),
        required=True,
        help="Package version type.",
    )
    parser.add_argument(
        "--version-file",
        required=True,
        help="Path to the VERSION_NUMBER file, relative to BUILD_SOURCESDIRECTORY.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    src_root = os.environ.get("BUILD_SOURCESDIRECTORY")
    if not src_root:
        print("##vso[task.logissue type=error]BUILD_SOURCESDIRECTORY is not set.")
        sys.exit(1)

    version_file = os.path.join(src_root, args.version_file)
    if not os.path.isfile(version_file):
        print(f"##vso[task.logissue type=error]Cannot find version number file at: {version_file}")
        sys.exit(1)

    with open(version_file) as f:
        original_ver = f.read().strip()

    if not original_ver:
        print("##vso[task.logissue type=error]VERSION_NUMBER is empty.")
        sys.exit(1)

    print(f"Original version: {original_ver}")
    print(f"Package version type: {args.package_version}")

    if args.package_version == "release":
        version_string = original_ver
        python_version = original_ver

    elif args.package_version == "rc":
        # RC versioning is not yet implemented. Fail the build to prevent publishing
        # an ambiguous version without an RC number.
        print("##vso[task.logissue type=error]RC versioning is not yet implemented. Use 'dev' or 'release' instead.")
        sys.exit(1)

    elif args.package_version == "dev":
        try:
            commit_sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short=8", "HEAD"],
                    cwd=src_root,
                )
                .decode("utf-8")
                .strip()
            )
            commit_timestamp = int(
                subprocess.check_output(
                    ["git", "show", "-s", "--format=%ct", "HEAD"],
                    cwd=src_root,
                )
                .decode("utf-8")
                .strip()
            )
            date_time_str = datetime.datetime.fromtimestamp(
                commit_timestamp,
                tz=datetime.timezone.utc,
            ).strftime("%Y%m%d%H%M%S")
        except Exception as e:
            print(f"##vso[task.logissue type=error]Failed to get git info: {e}")
            sys.exit(1)

        # The UTC commit timestamp determines dev-version precedence. Distinct commits made in the same second have
        # equal SemVer precedence and identical Python versions. If that is a problem, we can add a unique identifier
        # such as Azure Pipelines Build.BuildId to the precedence-bearing portion of each version.
        version_string = f"{original_ver}-dev.{date_time_str}+{commit_sha}"
        python_version = f"{original_ver}.dev{date_time_str}"

    print(f"Plugin package version string: {version_string}")
    print(f"Plugin Python package version string: {python_version}")

    # Validate semver 2.0.0 format
    semver_pattern = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    if not re.match(semver_pattern, version_string):
        print(f"##vso[task.logissue type=error]Version string '{version_string}' is not valid semver 2.0.0.")
        sys.exit(1)

    # Validate Python version (PEP 440)
    pep440_pattern = r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$"
    if not re.match(pep440_pattern, python_version):
        print(f"##vso[task.logissue type=error]Python version string '{python_version}' is not valid PEP 440.")
        sys.exit(1)

    print(f"##vso[task.setvariable variable=PluginPackageVersion]{version_string}")
    print(f"##vso[task.setvariable variable=PluginPythonPackageVersion]{python_version}")
    print(f"##vso[task.setvariable variable=PluginEpVersionDefine]onnxruntime_PLUGIN_EP_VERSION={version_string}")


if __name__ == "__main__":
    main()
