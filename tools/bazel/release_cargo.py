# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run the explicitly supplied installed-consumer Cargo executable."""

import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    tool = os.environ.get("ORT_RELEASE_CARGO")
    if not tool or not Path(tool).is_absolute() or not Path(tool).is_file():
        raise SystemExit(
            "Set --test_env=ORT_RELEASE_CARGO=/absolute/path/to/cargo to an installed Cargo executable; "
            "release consumers do not search PATH or skip missing tools."
        )
    sys.exit(subprocess.call([tool, *sys.argv[1:]]))
