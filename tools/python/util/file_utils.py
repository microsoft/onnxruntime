# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pathlib
import typing
import os


def path_match_suffix_ignore_case(path: typing.Union[pathlib.Path, str], suffix: str):
    '''TODO document'''
    if not isinstance(path, str):
        path = str(path)
    return path.casefold().endswith(suffix.casefold())


def files_from_file_or_dir(file_or_dir_path: typing.Union[pathlib.Path, str],
                           predicate: typing.Callable[[pathlib.Path], bool] = lambda _: True) \
        -> typing.List[pathlib.Path]:
    '''TODO document'''
    if not isinstance(file_or_dir_path, pathlib.Path):
        file_or_dir_path = pathlib.Path(file_or_dir_path)

    files = []

    def process_file(file_path: pathlib.Path):
        if predicate(file_path):
            files.append(file_path)

    if file_or_dir_path.is_dir():
        for root, _, files in os.walk(file_or_dir_path):
            for file in files:
                file_path = pathlib.Path(root, file)
                process_file(file_path)
    else:
        process_file(file_or_dir_path)

    return files
