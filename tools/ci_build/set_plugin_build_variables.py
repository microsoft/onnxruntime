#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Set plugin package version variables for Azure Pipelines.

Usage:
    python set_plugin_build_variables.py <package_version> <version_file_rel>

Where:
    package_version: 'release', 'RC', or 'dev'
    version_file_rel: path relative to BUILD_SOURCESDIRECTORY of the VERSION_NUMBER file
"""

import os
import re
import subprocess
import sys


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <package_version> <version_file_rel>")
        sys.exit(1)

    package_version = sys.argv[1]
    version_file_rel = sys.argv[2]

    if not version_file_rel:
        print("##vso[task.logissue type=error]version_file parameter is empty.")
        sys.exit(1)

    src_root = os.environ.get("BUILD_SOURCESDIRECTORY", "")
    version_file = os.path.join(src_root, version_file_rel)
    if not os.path.isfile(version_file):
        print(f"##vso[task.logissue type=error]Cannot find version number file at: {version_file}")
        sys.exit(1)

    with open(version_file) as f:
        original_ver = f.read().strip()

    if not original_ver:
        print("##vso[task.logissue type=error]VERSION_NUMBER is empty.")
        sys.exit(1)

    print(f"Original version: {original_ver}")
    print(f"Package version type: {package_version}")

    if package_version == "release":
        version_string = original_ver
        universal_version = original_ver
        python_version = original_ver

    elif package_version == "RC":
        # RC versioning is not yet implemented. Fail the build to prevent publishing
        # an ambiguous version without an RC number.
        print("##vso[task.logissue type=error]RC versioning is not yet implemented. Use 'dev' or 'release' instead.")
        sys.exit(1)

    elif package_version == "dev":
        try:
            commit_sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short=8", "HEAD"],
                    cwd=src_root,
                )
                .decode("utf-8")
                .strip()
            )
            date_str = (
                subprocess.check_output(
                    ["git", "show", "-s", "--format=%cd", "--date=format:%Y%m%d", "HEAD"],
                    cwd=src_root,
                )
                .decode("utf-8")
                .strip()
            )
        except Exception as e:
            print(f"##vso[task.logissue type=error]Failed to get git info: {e}")
            sys.exit(1)
        version_string = f"{original_ver}-dev.{date_str}+{commit_sha}"
        # Prefix the SHA with "commit-" so the pre-release identifier always contains a
        # non-digit. Otherwise, an all-numeric short SHA with a leading zero (e.g. "01234567")
        # would violate SemVer 2.0.0's rule against leading zeros in numeric identifiers.
        universal_version = f"{original_ver}-dev.{date_str}.commit-{commit_sha}"
        python_version = f"{original_ver}.dev{date_str}"

    else:
        print(
            f"##vso[task.logissue type=error]Unknown package_version '{package_version}'. Must be 'release', 'RC', or 'dev'."
        )
        sys.exit(1)

    print(f"Plugin package version string: {version_string}")
    print(f"Plugin universal package version string: {universal_version}")
    print(f"Plugin Python package version string: {python_version}")

    # Validate semver 2.0.0 format
    semver_pattern = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    if not re.match(semver_pattern, version_string):
        print(f"##vso[task.logissue type=error]Version string '{version_string}' is not valid semver 2.0.0.")
        sys.exit(1)

    # Validate universal version (SemVer 2.0.0, without build metadata)
    universal_semver_pattern = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?$"
    if not re.match(universal_semver_pattern, universal_version):
        print(
            f"##vso[task.logissue type=error]Universal version string '{universal_version}' is not valid semver 2.0.0 (without build metadata)."
        )
        sys.exit(1)

    # Validate Python version (PEP 440)
    pep440_pattern = r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$"
    if not re.match(pep440_pattern, python_version):
        print(f"##vso[task.logissue type=error]Python version string '{python_version}' is not valid PEP 440.")
        sys.exit(1)

    print(f"##vso[task.setvariable variable=PluginPackageVersion]{version_string}")
    print(f"##vso[task.setvariable variable=PluginUniversalPackageVersion]{universal_version}")
    print(f"##vso[task.setvariable variable=PluginPythonPackageVersion]{python_version}")
    print(f"##vso[task.setvariable variable=PluginEpVersionDefine]onnxruntime_PLUGIN_EP_VERSION={version_string}")


if __name__ == "__main__":
    main()
