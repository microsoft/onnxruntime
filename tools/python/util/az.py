# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import logging
import subprocess


_log = logging.getLogger("util.az")


def az(*args, az_path="az", cwd=None, parse_output=True, quiet=False):
    """Runs an Azure CLI command ("az ...") and optionally returns the parsed
        JSON output.

    Args:
        *args: The Azure CLI command without the leading "az".
        az_path: The path to the az client.
        cwd: The working directory. If None, specifies the current directory.
        parse_output: Whether to parse the JSON output.
        quiet: Whether to suppress printed output.

    Returns:
        The parsed JSON output of the command if desired.
    """
    cmd = [az_path, *args, "--output", "json"]
    default_output = subprocess.DEVNULL if quiet else None
    stdout = subprocess.PIPE if parse_output else default_output
    if not quiet:
        _log.debug("Running command{}: {}".format(
            " in '{}'".format(cwd) if cwd is not None else "", cmd))
    process = subprocess.run(
        cmd, cwd=cwd, stdout=stdout, stderr=default_output, universal_newlines=True, check=True)
    return json.loads(process.stdout) if parse_output else None
