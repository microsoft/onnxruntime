# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import pathlib
import re
import shutil


_script_dir = pathlib.Path(__file__).parent.resolve(strict=True)
repo_root = _script_dir.parents[3]

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
    :param variable_substitutions The mapping from template variable name to value.
    '''
    with open(template_file, mode="r") as template:
        content = template.read()

    def replace_template_variable(match):
        variable_name = match.group(1)
        return variable_substitutions.get(variable_name, match.group(0))

    content = _template_variable_pattern.sub(replace_template_variable, content)

    with open(output_file, mode="w") as output:
        output.write(content)


def copy_repo_relative_to_dir(patterns: list[str], dest_dir: pathlib.Path):
    '''
    Copies file paths relative to the repo root to a directory.
    The given paths or path patterns are relative to the repo root, and the
    repo root-relative intermediate directory structure is maintained.

    :param patterns The paths or path patterns relative to the repo root.
    :param dest_dir The destination directory.
    '''
    paths = [path for pattern in patterns for path in repo_root.glob(pattern)]
    for path in paths:
        repo_relative_path = path.relative_to(repo_root)
        dst_path = dest_dir / repo_relative_path
        os.makedirs(dst_path.parent, exist_ok=True)
        shutil.copy(path, dst_path)


def load_framework_info(framework_info_file: pathlib.Path):
    '''
    Loads framework info from a file.

    :param framework_info_file The framework info file path.
    :return The framework info values.
    '''
    with open(framework_info_file, mode="r") as framework_info:
        return json.load(framework_info)
