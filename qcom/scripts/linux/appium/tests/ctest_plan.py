# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import re
from pathlib import Path


class CTestPlan:
    def __init__(self, ctesttestfile_path: Path, device_test_root: str):
        """
        Create a new instance
        :param ctesttestfile_path: Path to CTestTestfile.cmake on this host machine.
        :param device_test_root:
        """
        if not ctesttestfile_path.is_file():
            raise FileExistsError(f"CTestTestfile {ctesttestfile_path} does not exist.")
        build_directory = self.__parse_build_directory(ctesttestfile_path)
        self.__tests = self.__parse_tests(ctesttestfile_path, build_directory, device_test_root)

    @property
    def plan(self) -> list[list[str]]:
        return list(self.__tests.values())

    @staticmethod
    def __parse_build_directory(ctesttestfile_path: Path) -> str:
        build_dir_re = re.compile(r"^# Build directory: (.*)$")
        for line in ctesttestfile_path.read_text().splitlines():
            matches = build_dir_re.match(line)
            if matches:
                return matches.group(1)
        raise ValueError(f"Could not find build directory in {ctesttestfile_path}.")

    def __parse_tests(
        self, ctesttestfile_path: Path, build_directory: str, device_test_root: str
    ) -> dict[str, list[str]]:
        tests = {}
        # This isn't technically correct since it doesn't allow for double quotes embedded in paths, but we'll
        # accept the shortcoming due to its improbability.
        add_test_re = re.compile(r'add_test\(\[=\[(.*)\]=\] "([^"]*)"(?: "([^"]*)")*\)')

        for line in ctesttestfile_path.read_text().splitlines():
            matches = add_test_re.match(line)
            if matches:
                test_name = matches.group(1).replace(build_directory, device_test_root)
                cmd = [
                    # CTestTestfiles.cmake produced on Windows uses both / and \\ as path separators.
                    m.replace("\\\\", "/").replace(build_directory, device_test_root)
                    for m in matches.groups()[1:]
                ]
                tests[test_name] = cmd
        return tests
