# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by "Python Packaging Pipeline (Training Cuda 11.6)", "Python Packaging Pipeline - No Local Version (Training Cuda 11.6)", "Python packaging pipeline".
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
REPO_DIR = SCRIPT_DIR.parent.parent

with (REPO_DIR / ".config" / "tsaoptions.json").open() as f:
    data = json.load(f)

buildNumber = os.getenv("BUILD_BUILDNUMBER")
if buildNumber is not None:
    data["buildNumber"] = buildNumber

with (REPO_DIR / ".config" / "tsaoptions.json").open(mode="w") as f:
    json.dump(data, f)
