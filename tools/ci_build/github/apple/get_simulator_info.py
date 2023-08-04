#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import enum
import json
import subprocess

from packaging.version import Version


class SimulatorInfoKey(str, enum.Enum):
    """
    The type of simulator information that can be retrieved by get_simulator_info().
    """

    DeviceTypeIdentifier = "device_type_identifier"
    DeviceTypeName = "device_type_name"
    RuntimeIdentifier = "runtime_identifier"
    RuntimePlatform = "runtime_platform"
    RuntimeVersion = "runtime_version"

    def __str__(self):
        return self.value


def get_simulator_info(
    requested_runtime_platform: str | None = "iOS",
    requested_device_type_product_family: str | None = "iPhone",
    max_runtime_version: str | None = "16.4",
) -> dict[str, str]:
    """
    Retrieves simulator information for a runtime and device type from Xcode.
    This specifies a simulator configuration appropriate for running tests on this machine.

    :param requested_runtime_platform: The runtime platform to select.
    :param requested_device_type_product_family: The device type product family to select.
    :param max_runtime_version: The maximum runtime version to allow.

    :return: A dictionary containing the simulator information for the selected runtime and device type.
             The keys are specified by the SimulatorInfoKey enum.
    """
    simctl_proc = subprocess.run(
        ["xcrun", "simctl", "list", "--json", "--no-escape-slashes"],
        text=True,
        capture_output=True,
        check=True,
    )

    simctl_json = json.loads(simctl_proc.stdout)

    # choose runtime - pick the one with the largest version not greater than max_runtime_version
    def runtime_filter(runtime) -> bool:
        if max_runtime_version is not None and Version(runtime["version"]) > Version(max_runtime_version):
            return False

        if requested_runtime_platform is not None and runtime["platform"] != requested_runtime_platform:
            return False

        return True

    selected_runtime = max(
        filter(runtime_filter, simctl_json["runtimes"]),
        key=lambda runtime: Version(runtime["version"]),
    )

    # choose device type - pick the one with the largest minimum runtime version that is supported by selected_runtime
    selected_runtime_supported_device_type_ids = set(
        device_type["identifier"] for device_type in selected_runtime["supportedDeviceTypes"]
    )

    def device_type_filter(device_type) -> bool:
        if (
            requested_device_type_product_family is not None
            and device_type["productFamily"] != requested_device_type_product_family
        ):
            return False

        if device_type["identifier"] not in selected_runtime_supported_device_type_ids:
            return False

        return True

    selected_device_type = max(
        filter(device_type_filter, simctl_json["devicetypes"]),
        key=lambda device_type: device_type["minRuntimeVersion"],
    )

    result = {
        str(SimulatorInfoKey.DeviceTypeIdentifier): selected_device_type["identifier"],
        str(SimulatorInfoKey.DeviceTypeName): selected_device_type["name"],
        str(SimulatorInfoKey.RuntimeIdentifier): selected_runtime["identifier"],
        str(SimulatorInfoKey.RuntimePlatform): selected_runtime["platform"],
        str(SimulatorInfoKey.RuntimeVersion): selected_runtime["version"],
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Gets simulator info from Xcode.")
    parser.add_argument(
        "output_format",
        nargs="?",
        help="Specifies the output format. If unspecified, all info is printed in JSON format. This should be a "
        "format string compatible with Python `str.format_map()`. "
        f"Possible replacement field names are: {[str(key) for key in SimulatorInfoKey]}. "
        f'Example: "OS={{{SimulatorInfoKey.RuntimeVersion}}}".',
    )
    args = parser.parse_args()

    info = get_simulator_info()

    if args.output_format is not None:
        print(args.output_format.format_map(info))
    else:
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
