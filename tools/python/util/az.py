# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import subprocess


def az(*args, az_path="az", cwd=None, parse_output=True):
    """Runs an Azure CLI command ("az ...") and optionally returns the parsed
        JSON output.

    Args:
        *args: The Azure CLI command without the leading "az".
        az_path: The path to the az client.
        cwd: The working directory. If None, specifies the current directory.
        parse_output: Whether to parse the JSON output.

    Returns:
        The parsed JSON output of the command if desired.
    """
    cmd = [az_path, *args, "--output", "json"]
    stdout = subprocess.PIPE if parse_output else None
    process = subprocess.run(
        cmd, cwd=cwd, stdout=stdout, universal_newlines=True, check=True)
    return json.loads(process.stdout) if parse_output else None
