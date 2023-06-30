# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import enum
import json
import os
import pathlib
import re
import shutil
from typing import Dict, List

_script_dir = pathlib.Path(__file__).parent.resolve(strict=True)
repo_root = _script_dir.parents[3]


class PackageVariant(enum.Enum):
    Full = 0  # full ORT build with all opsets, ops, and types
    Mobile = 1  # minimal ORT build with reduced ops
    Training = 2  # full ORT build with all opsets, ops, and types, plus training APIs
    Test = -1  # for testing purposes only

    @classmethod
    def release_variant_names(cls):
        return [v.name for v in cls if v.value >= 0]

    @classmethod
    def all_variant_names(cls):
        return [v.name for v in cls]


_template_variable_pattern = re.compile(r"@(\w+)@")  # match "@var@"


def gen_file_from_template(
    template_file: pathlib.Path, output_file: pathlib.Path, variable_substitutions: Dict[str, str], strict: bool = True
):
    """
    Generates a file from a template file.
    The template file may contain template variables that will be substituted
    with the provided values in the generated output file.
    In the template file, template variable names are delimited by "@"'s,
    e.g., "@var@".

    :param template_file The template file path.
    :param output_file The generated output file path.
    :param variable_substitutions The mapping from template variable name to value.
    :param strict Whether to require the set of template variable names in the file and the keys of
                  `variable_substitutions` to be equal.
    """
    with open(template_file) as template:
        content = template.read()

    variables_in_file = set()

    def replace_template_variable(match):
        variable_name = match.group(1)
        variables_in_file.add(variable_name)
        return variable_substitutions.get(variable_name, match.group(0))

    content = _template_variable_pattern.sub(replace_template_variable, content)

    if strict and variables_in_file != variable_substitutions.keys():
        variables_in_substitutions = set(variable_substitutions.keys())
        raise ValueError(
            f"Template file variables and substitution variables do not match. "
            f"Only in template file: {sorted(variables_in_file - variables_in_substitutions)}. "
            f"Only in substitutions: {sorted(variables_in_substitutions - variables_in_file)}."
        )

    with open(output_file, mode="w") as output:
        output.write(content)


def filter_files(all_file_patterns: List[str], excluded_file_patterns: List[str]):
    """
    Filters file paths based on inclusion and exclusion patterns

    :param all_file_patterns The list of file paths to filter.
    :param excluded_file_patterns The list of exclusion patterns.

    :return The filtered list of file paths
    """
    # get all files matching the patterns in all_file_patterns
    all_files = [str(path.relative_to(repo_root)) for pattern in all_file_patterns for path in repo_root.glob(pattern)]

    # get all files matching the patterns in excluded_file_patterns
    exclude_files = [
        str(path.relative_to(repo_root)) for pattern in excluded_file_patterns for path in repo_root.glob(pattern)
    ]

    # return the difference
    return list(set(all_files) - set(exclude_files))


def copy_repo_relative_to_dir(patterns: List[str], dest_dir: pathlib.Path):
    """
    Copies file paths relative to the repo root to a directory.
    The given paths or path patterns are relative to the repo root, and the
    repo root-relative intermediate directory structure is maintained.

    :param patterns The paths or path patterns relative to the repo root.
    :param dest_dir The destination directory.
    """
    paths = [path for pattern in patterns for path in repo_root.glob(pattern)]
    for path in paths:
        repo_relative_path = path.relative_to(repo_root)
        dst_path = dest_dir / repo_relative_path
        os.makedirs(dst_path.parent, exist_ok=True)
        shutil.copy(path, dst_path)


def load_json_config(json_config_file: pathlib.Path):
    """
    Loads configuration info from a JSON file.

    :param json_config_file The JSON configuration file path.
    :return The configuration info values.
    """
    with open(json_config_file) as config:
        return json.load(config)


def get_ort_version():
    """
    Gets the ONNX Runtime version string from the repo.

    :return The ONNX Runtime version string.
    """
    with open(repo_root / "VERSION_NUMBER") as version_file:
        return version_file.read().strip()
