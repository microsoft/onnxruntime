#!/usr/bin/env python3

# args: <submodule path>
# output: <path> <url> <commit>

import subprocess
import sys

assert len(sys.argv) == 2

path = sys.argv[1]

proc = subprocess.run(
    ["git", "config", "--get", "remote.origin.url"],
    check=True,
    cwd=path,
    stdout=subprocess.PIPE,
    text=True,
)

url = proc.stdout.strip()

proc = subprocess.run(["git", "rev-parse", "HEAD"], check=True, cwd=path, stdout=subprocess.PIPE, text=True)

commit = proc.stdout.strip()

print(f"{path} {url} {commit}")
