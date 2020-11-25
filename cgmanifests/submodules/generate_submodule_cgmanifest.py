#!/usr/bin/env python3

import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

proc = subprocess.run(
    ["git", "submodule", "foreach", "--quiet", "--recursive", "{} {} $toplevel/$sm_path".format(
        sys.executable, os.path.join(SCRIPT_DIR, "print_submodule_info.py"))],
    check=True,
    cwd=REPO_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True)

registrations = []
submodule_lines = proc.stdout.splitlines()
for submodule_line in submodule_lines:
    (absolute_path, url, commit) = submodule_line.split(" ")
    registration = {
        "component": {
            "type": "git",
            "git": {
                "commitHash": commit,
                "repositoryUrl": url,
            },
            "comments": "git submodule at {}".format(os.path.relpath(absolute_path, REPO_DIR))
        }
    }
    registrations.append(registration)

cgmanifest = {"Version": 1, "Registrations": registrations}

print(json.dumps(cgmanifest, indent=2))
