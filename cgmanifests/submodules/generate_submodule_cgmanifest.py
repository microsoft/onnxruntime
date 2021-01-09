#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import re

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

package_name = None
package_filename = None
package_url = None

registrations = []

with open(os.path.join(REPO_DIR,'tools','ci_build','github','linux','docker','manylinux2014_build_scripts','build_env.sh'), "r") as f:
  for line in f:
    if not line.strip():      
      package_name = None
      package_filename = None
      package_url = None
    if package_filename is None:
      m = re.match("(.+?)_ROOT=(.*)$",line)
      if m != None:
        package_name = m.group(1)
        package_filename = m.group(2)
      else:
        m = re.match("(.+?)_AUTOCONF_VERSION=(.*)$",line)
        if m != None:
          package_name = m.group(1)
          package_filename = m.group(2)        
    elif package_url is None:
      m = re.match("(.+?)_DOWNLOAD_URL=(.+)$",line)
      if m != None:
        package_url = m.group(2) + "/" + package_filename + ".tar.gz"
        registration = {
          "Component": {
            "Type": "other",
            "other": {
                "Name": package_name.lower(),
                "Version": package_filename.split("-")[-1],
                "DownloadUrl": package_url,
            },
            "comments": "manylinux dependency"
          }
        }
        registrations.append(registration)
        package_name = None
        package_filename = None
        package_url = None


proc = subprocess.run(
    ["git", "submodule", "foreach", "--quiet", "--recursive", "{} {} $toplevel/$sm_path".format(
        sys.executable, os.path.join(SCRIPT_DIR, "print_submodule_info.py"))],
    check=True,
    cwd=REPO_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True)


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
