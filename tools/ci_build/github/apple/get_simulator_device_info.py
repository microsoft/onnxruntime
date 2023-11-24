#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import functools
import itertools
import json
import subprocess


@functools.total_ordering
class Version:
    """
    A simple Version class.
    We opt to use this instead of `packaging.version.Version` to avoid depending on the external `packaging` package.
    It only supports integer version components.
    """

    def __init__(self, version_string: str):
        self._components = tuple(int(component) for component in version_string.split("."))

    def __eq__(self, other: Version) -> bool:
        component_pairs = itertools.zip_longest(self._components, other._components, fillvalue=0)
        return all(pair[0] == pair[1] for pair in component_pairs)

    def __lt__(self, other: Version) -> bool:
        component_pairs = itertools.zip_longest(self._components, other._components, fillvalue=0)
        for self_component, other_component in component_pairs:
            if self_component != other_component:
                return self_component < other_component
        return False


def get_simulator_device_info(
    requested_runtime_platform: str = "iOS",
    requested_device_type_product_family: str = "iPhone",
    max_runtime_version_str: str | None = None,
) -> dict[str, str]:
    """
    Retrieves simulator device information from Xcode.
    This simulator device should be appropriate for running tests on this machine.

    :param requested_runtime_platform: The runtime platform to select.
    :param requested_device_type_product_family: The device type product family to select.
    :param max_runtime_version_str: The maximum runtime version to allow.

    :return: A dictionary containing information about the selected simulator device.
    """
    max_runtime_version = Version(max_runtime_version_str) if max_runtime_version_str is not None else None

    simctl_proc = subprocess.run(
        ["xcrun", "simctl", "list", "--json", "--no-escape-slashes"],
        text=True,
        capture_output=True,
        check=True,
    )

    simctl_json = json.loads(simctl_proc.stdout)

    # device type id -> device type structure
    device_type_map = {device_type["identifier"]: device_type for device_type in simctl_json["devicetypes"]}

    # runtime id -> runtime structure
    runtime_map = {runtime["identifier"]: runtime for runtime in simctl_json["runtimes"]}

    def runtime_filter(runtime) -> bool:
        if not runtime["isAvailable"]:
            return False

        if runtime["platform"] != requested_runtime_platform:
            return False

        if max_runtime_version is not None and Version(runtime["version"]) > max_runtime_version:
            return False

        return True

    def runtime_id_filter(runtime_id: str) -> bool:
        runtime = runtime_map.get(runtime_id)
        if runtime is None:
            return False
        return runtime_filter(runtime)

    def device_type_filter(device_type) -> bool:
        if device_type["productFamily"] != requested_device_type_product_family:
            return False

        return True

    def device_filter(device) -> bool:
        if not device["isAvailable"]:
            return False

        if not device_type_filter(device_type_map[device["deviceTypeIdentifier"]]):
            return False

        return True

    # simctl_json["devices"] is a map of runtime id -> list of device structures
    # expand this into a list of (runtime id, device structure) and filter out invalid entries
    runtime_id_and_device_pairs = []
    for runtime_id, device_list in filter(
        lambda runtime_id_and_device_list: runtime_id_filter(runtime_id_and_device_list[0]),
        simctl_json["devices"].items(),
    ):
        runtime_id_and_device_pairs.extend((runtime_id, device) for device in filter(device_filter, device_list))

    # sort key - tuple of (runtime version, device type min runtime version)
    # the secondary device type min runtime version value is to treat more recent device types as greater
    def runtime_id_and_device_pair_key(runtime_id_and_device_pair):
        runtime_id, device = runtime_id_and_device_pair

        runtime = runtime_map[runtime_id]
        device_type = device_type_map[device["deviceTypeIdentifier"]]

        return (Version(runtime["version"]), device_type["minRuntimeVersion"])

    selected_runtime_id, selected_device = max(runtime_id_and_device_pairs, key=runtime_id_and_device_pair_key)
    selected_runtime = runtime_map[selected_runtime_id]
    selected_device_type = device_type_map[selected_device["deviceTypeIdentifier"]]

    result = {
        "device_name": selected_device["name"],
        "device_udid": selected_device["udid"],
        "device_type_identifier": selected_device_type["identifier"],
        "device_type_name": selected_device_type["name"],
        "device_type_product_family": selected_device_type["productFamily"],
        "runtime_identifier": selected_runtime["identifier"],
        "runtime_platform": selected_runtime["platform"],
        "runtime_version": selected_runtime["version"],
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Gets simulator info from Xcode and prints it in JSON format.")
    _ = parser.parse_args()  # no args yet

    info = get_simulator_device_info(
        # The macOS-13 hosted agent image has iOS 17 which is currently in beta. Limit it to 16.4 for now.
        # See https://github.com/actions/runner-images/issues/8023
        # TODO Remove max_runtime_version limit.
        max_runtime_version_str="16.4",
    )

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
