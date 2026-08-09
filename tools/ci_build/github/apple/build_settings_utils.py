from __future__ import annotations

import json
import pathlib
import typing

_DEFAULT_BUILD_SYSROOT_ARCHS = {
    "iphoneos": ["arm64"],
    "iphonesimulator": ["arm64", "x86_64"],
}


def parse_build_settings_file(build_settings_file: pathlib.Path) -> dict[str, typing.Any]:
    """
    Parses the provided build settings file into a build settings dict.

    :param build_settings_file: The build settings file path.
    :type build_settings_file: pathlib.Path
    :return: The build settings dict.
    :rtype: dict[str, Any]
    """

    def check(condition: bool, message: str):
        if not condition:
            raise ValueError(message)

    # validate that `input` is a dict[str, list[str]]
    def validate_str_to_str_list_dict(input: dict[str, list[str]]):
        check(isinstance(input, dict), f"input is not a dict: {input}")
        for key, value in input.items():
            check(isinstance(key, str), f"key is not a string: {key}")
            check(isinstance(value, list), f"value is not a list: {value}")
            for value_element in value:
                check(isinstance(value_element, str), f"list element is not a string: {value_element}")

    with open(build_settings_file) as f:
        build_settings_data = json.load(f)

    build_settings = {}

    build_osx_archs = build_settings_data.get("build_osx_archs", _DEFAULT_BUILD_SYSROOT_ARCHS)
    validate_str_to_str_list_dict(build_osx_archs)
    build_settings["build_osx_archs"] = build_osx_archs

    build_params = build_settings_data.get("build_params", {})
    validate_str_to_str_list_dict(build_params)
    build_settings["build_params"] = build_params

    return build_settings


def get_sysroot_arch_pairs(build_settings: dict) -> list[tuple[str, str]]:
    """
    Gets all specified sysroot/arch pairs.

    :param build_settings: The build settings dict.
    :type build_settings: dict
    :return: A list of (sysroot, arch) tuples.
    :rtype: list[tuple[str, str]]
    """
    pair_set: set[tuple[str, str]] = set()
    for sysroot, archs in build_settings["build_osx_archs"].items():
        for arch in archs:
            pair_set.add((sysroot, arch))

    return sorted(pair_set)


def get_build_params(build_settings: dict, sysroot: str) -> list[str]:
    """
    Returns the build params associated with given `sysroot`.
    The special `sysroot` value "base" may be used to get the base build params.

    :param build_settings: The build settings dict.
    :type build_settings: dict
    :param sysroot: The specified sysroot.
    :type sysroot: str
    :return: The build params associated with `sysroot`, if any, or an empty list.
    :rtype: list[str]
    """
    return build_settings["build_params"].get(sysroot, [])
