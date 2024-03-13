#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse

import requests


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Github username")
    parser.add_argument("--token", required=True, help="Github access token")

    return parser.parse_args()


args = parse_arguments()

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

package_name = None
package_filename = None
package_url = None

registrations = []


@dataclass(frozen=True)
class GitDep:
    commit: str
    url: str


git_deps = {}


def add_github_dep(name, parsed_url):
    segments = parsed_url.path.split("/")
    org_name = segments[1]
    repo_name = segments[2]
    if segments[3] != "archive":
        print("unrecognized github url path:" + parsed_url.path)
        return
    git_repo_url = f"https://github.com/{org_name}/{repo_name}.git"
    # For example, the path might be like '/myorg/myrepo/archive/5a5f8a5935762397aa68429b5493084ff970f774.zip'
    # The last segment, segments[4], is '5a5f8a5935762397aa68429b5493084ff970f774.zip'
    if len(segments) == 5 and re.match(r"[0-9a-f]{40}", PurePosixPath(segments[4]).stem):
        commit = PurePosixPath(segments[4]).stem
        dep = GitDep(commit, git_repo_url)
        if dep not in git_deps:
            git_deps[dep] = name
    else:
        # TODO: support urls like: https://github.com/onnx/onnx-tensorrt/archive/refs/tags/release/7.1.zip
        if len(segments) == 5:
            tag = PurePosixPath(segments[4]).stem
            if tag.endswith(".tar"):
                tag = PurePosixPath(tag).stem
        elif segments[4] == "refs" and segments[5] == "tags":
            tag = PurePosixPath(segments[6]).stem
            if tag.endswith(".tar"):
                tag = PurePosixPath(tag).stem
        else:
            print("unrecognized github url path:" + parsed_url.path)
            return
        # Make a REST call to convert to tag to a git commit
        url = f"https://api.github.com/repos/{org_name}/{repo_name}/git/refs/tags/{tag}"
        print("requesting %s ..." % url)
        res = requests.get(url, auth=(args.username, args.token))
        response_json = res.json()
        tag_object = response_json["object"]
        if tag_object["type"] == "commit":
            commit = tag_object["sha"]
        elif tag_object["type"] == "tag":
            res = requests.get(tag_object["url"], auth=(args.username, args.token))
            commit = res.json()["object"]["sha"]
        else:
            print("unrecognized github url path:" + parsed_url.path)
            return
        dep = GitDep(commit, git_repo_url)
        if dep not in git_deps:
            git_deps[dep] = name


def normalize_path_separators(path):
    return path.replace(os.path.sep, "/")


proc = subprocess.run(
    [
        "git",
        "submodule",
        "foreach",
        "--quiet",
        "'{}' '{}' $toplevel/$sm_path".format(
            normalize_path_separators(sys.executable),
            normalize_path_separators(os.path.join(SCRIPT_DIR, "print_submodule_info.py")),
        ),
    ],
    check=True,
    cwd=REPO_DIR,
    capture_output=True,
    text=True,
)


submodule_lines = proc.stdout.splitlines()
for submodule_line in submodule_lines:
    (absolute_path, url, commit) = submodule_line.split(" ")
    git_deps[GitDep(commit, url)] = (
        f"git submodule at {normalize_path_separators(os.path.relpath(absolute_path, REPO_DIR))}"
    )

with open(os.path.join(SCRIPT_DIR, "..", "cmake", "deps.txt")) as f:
    depfile_reader = csv.reader(f, delimiter=";")
    for row in depfile_reader:
        if len(row) < 3:
            continue
        name = row[0]
        # Lines start with "#" are comments
        if name.startswith("#"):
            continue
        url = row[1]
        parsed_url = urlparse(url)
        # TODO: add support for gitlab
        if parsed_url.hostname == "github.com":
            add_github_dep(name, parsed_url)
        else:
            print("unrecognized url:" + url)

for git_dep, comment in git_deps.items():
    registration = {
        "component": {
            "type": "git",
            "git": {
                "commitHash": git_dep.commit,
                "repositoryUrl": git_dep.url,
            },
            "comments": comment,
        }
    }
    registrations.append(registration)

cgmanifest = {
    "$schema": "https://json.schemastore.org/component-detection-manifest.json",
    "Version": 1,
    "Registrations": registrations,
}

with open(os.path.join(SCRIPT_DIR, "generated", "cgmanifest.json"), mode="w") as generated_cgmanifest_file:
    print(json.dumps(cgmanifest, indent=2), file=generated_cgmanifest_file)
