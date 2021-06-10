#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pathlib
import re
import shutil
import sys


_script_dir = pathlib.Path(__file__).parent.resolve(strict=True)
_repo_root = _script_dir.parents[4]

# these variables contain paths or path patterns that are relative to the repo root

# the license file
license_file = "LICENSE"

# include directories for compiling the pod itself
include_dirs = [
    "objectivec",
    "cmake/external/SafeInt",
]

# pod source files
source_files = [
    "objectivec/include/*.h",
    "objectivec/src/*.h",
    "objectivec/src/*.m",
    "objectivec/src/*.mm",
    "cmake/external/SafeInt/safeint/SafeInt.hpp",
]

# pod public header files
# note: these are a subset of source_files
public_header_files = [
    "objectivec/include/*.h",
]

# pod test source files
test_source_files = [
    "objectivec/test/*.h",
    "objectivec/test/*.m",
    "objectivec/test/*.mm",
]

# pod test resource files
test_resource_files = [
    "objectivec/test/testdata/*.ort",
]


def parse_args():
    parser = argparse.ArgumentParser(description="""
        Assembles the files for the Objective-C pod package in a staging directory.
        This directory can be validated (e.g., with `pod lib lint`) and then zipped to create a package for release.
    """)

    parser.add_argument("--staging-dir", type=pathlib.Path,
                        default=pathlib.Path("./onnxruntime-mobile-objc-staging"),
                        help="Staging directory for the Objective-C pod files.")
    parser.add_argument("--pod-version", required=True,
                        help="Objective-C pod version.")
    parser.add_argument("--pod-source-http", required=True,
                        help="Objective-C pod source http value (e.g., the package download URL).")

    return parser.parse_args()


_template_variable_pattern = re.compile(r"@(\w+)@")  # match "@var@"


def gen_file_from_template(template_file: pathlib.Path, output_file: pathlib.Path,
                           variable_substitutions: dict[str, str]):
    '''
    Generates a file from a template file.
    The template file may contain template variables that will be substituted
    with the provided values in the generated output file.
    In the template file, template variable names are delimited by "@"'s,
    e.g., "@var@".

    :param template_file The template file path.
    :param output_file The generated output file path.
    :variable_substitutions The mapping from template variable name to value.
    '''
    with open(template_file, mode="r") as template:
        content = template.read()

    def replace_template_variable(match):
        variable_name = match.group(1)
        return variable_substitutions.get(variable_name, match.group(0))

    content = _template_variable_pattern.sub(replace_template_variable, content)

    with open(output_file, mode="w") as output:
        output.write(content)


def copy_to_staging_dir(patterns: list[str], staging_dir: pathlib.Path):
    '''
    Copies files to a staging directory.
    The files are relative to the repo root, and the repo root-relative
    intermediate directory structure is maintained.

    :param patterns The paths or path patterns relative to the repo root.
    :param staging_dir The staging directory.
    '''
    paths = [path for pattern in patterns for path in _repo_root.glob(pattern)]
    for path in paths:
        repo_relative_path = path.relative_to(_repo_root)
        dst_path = staging_dir / repo_relative_path
        os.makedirs(dst_path.parent, exist_ok=True)
        shutil.copy(path, dst_path)


def main():
    args = parse_args()

    staging_dir = args.staging_dir.resolve()
    print(f"Assembling files in staging directory: {staging_dir}")
    if staging_dir.exists():
        print("Warning: staging directory already exists", file=sys.stderr)

    # copy the necessary files to the staging directory
    copy_to_staging_dir(
        [license_file] + source_files + test_source_files + test_resource_files,
        staging_dir)

    # generate the podspec file from the template

    def path_patterns_as_variable_value(patterns: list[str]):
        return ", ".join([f'"{pattern}"' for pattern in patterns])

    variable_substitutions = {
        "VERSION": args.pod_version,
        "SOURCE_HTTP": args.pod_source_http,
        "LICENSE_FILE": path_patterns_as_variable_value([license_file]),
        "INCLUDE_DIR_LIST": path_patterns_as_variable_value(include_dirs),
        "PUBLIC_HEADER_FILE_LIST": path_patterns_as_variable_value(public_header_files),
        "SOURCE_FILE_LIST": path_patterns_as_variable_value(source_files),
        "TEST_SOURCE_FILE_LIST": path_patterns_as_variable_value(test_source_files),
        "TEST_RESOURCE_FILE_LIST": path_patterns_as_variable_value(test_resource_files),
    }

    podspec_template = _script_dir / "onnxruntime-mobile-objc.podspec.template"
    podspec = staging_dir / "onnxruntime-mobile-objc.podspec"

    gen_file_from_template(podspec_template, podspec, variable_substitutions)

    return 0


if __name__ == "__main__":
    sys.exit(main())
